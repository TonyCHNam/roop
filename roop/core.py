#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import onnxruntime
import tensorflow
import roop.globals
import roop.metadata
import roop.ui as ui
from roop.predictor import predict_image, predict_video
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import (
    has_image_extension, is_image, is_video, detect_fps, create_video,
    extract_frames, get_temp_frame_paths, restore_audio, create_temp,
    move_temp, clean_temp, normalize_output_path
)

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='frame processors (choices: face_swapper, face_enhancer, ...)', dest='frame_processor', default=['face_swapper'], nargs='+')
    program.add_argument('--keep-fps', help='keep target fps', dest='keep_fps', action='store_true')
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true')
    program.add_argument('--skip-audio', help='skip target audio', dest='skip_audio', action='store_true')
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true')
    program.add_argument('--reference-face-position', help='position of the reference face', dest='reference_face_position', type=int, default=0)
    program.add_argument('--reference-frame-number', help='number of the reference frame', dest='reference_frame_number', type=int, default=0)
    program.add_argument('--similar-face-distance', help='face distance used for recognition', dest='similar_face_distance', type=float, default=0.85)
    program.add_argument('--temp-frame-format', help='image format used for frame extraction', dest='temp_frame_format', default='png', choices=['jpg', 'png'])
    program.add_argument('--temp-frame-quality', help='image quality used for frame extraction', dest='temp_frame_quality', type=int, default=0, choices=range(101), metavar='[0-100]')
    program.add_argument('--output-video-encoder', help='encoder used for the output video', dest='output_video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc'])
    program.add_argument('--output-video-quality', help='quality used for the output video', dest='output_video_quality', type=int, default=35, choices=range(101), metavar='[0-100]')
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int)
    program.add_argument('--execution-provider', help='available execution provider (choices: cpu, ...)', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{roop.metadata.name} {roop.metadata.version}')

    args = program.parse_args()

    roop.globals.source_path = args.source_path
    roop.globals.target_path = args.target_path
    roop.globals.output_path = normalize_output_path(roop.globals.source_path, roop.globals.target_path, args.output_path)
    roop.globals.headless = roop.globals.source_path is not None and roop.globals.target_path is not None and roop.globals.output_path is not None
    roop.globals.frame_processors = args.frame_processor
    roop.globals.keep_fps = args.keep_fps
    roop.globals.keep_frames = args.keep_frames
    roop.globals.skip_audio = args.skip_audio
    roop.globals.many_faces = args.many_faces
    roop.globals.reference_face_position = args.reference_face_position
    roop.globals.reference_frame_number = args.reference_frame_number
    roop.globals.similar_face_distance = args.similar_face_distance
    roop.globals.temp_frame_format = args.temp_frame_format
    roop.globals.temp_frame_quality = args.temp_frame_quality
    roop.globals.output_video_encoder = args.output_video_encoder
    roop.globals.output_video_quality = args.output_video_quality
    roop.globals.max_memory = args.max_memory
    roop.globals.execution_providers = decode_execution_providers(args.execution_provider)
    roop.globals.execution_threads = args.execution_threads


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        return 8
    return 1


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
            tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
        ])
    # limit memory usage
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'ROOP.CORE') -> None:
    print(f'[{scope}] {message}')
    if not roop.globals.headless:
        ui.update_status(message)


def start() -> None:
    # 기존 프레임 추출 및 영상 생성 단계 (필요한 경우 그대로 유지)
    # 1. SimSwap 기반 얼굴 스왑 진행
    initial_output = roop.globals.output_path  # 예: output.mp4
    print("SimSwap 처리 중...")
    process_video_with_simswap(roop.globals.source_path, roop.globals.target_path, initial_output)

    # 2. GFPGAN을 통한 얼굴 디테일 향상
    gfpgan_output = initial_output.replace('.mp4', '_enhanced.mp4')
    print("GFPGAN 처리 중...")
    enhance_faces_gfpgan(initial_output, gfpgan_output)

    # 3. FOMM을 통한 얼굴 움직임 및 표정 보정
    fomm_output = gfpgan_output.replace('.mp4', '_motion.mp4')
    print("FOMM 처리 중...")
    apply_fomm(roop.globals.source_path, gfpgan_output, fomm_output)

    # 4. Real-ESRGAN을 통한 해상도 업스케일
    final_output = fomm_output.replace('.mp4', '_4K.mp4')
    print("Real-ESRGAN 처리 중...")
    enhance_video_with_realesrgan(fomm_output, final_output, scale=2, device='cuda')

    print(f"최종 고품질 비디오가 {final_output} 에 저장되었습니다.")


def destroy() -> None:
    if roop.globals.target_path:
        clean_temp(roop.globals.target_path)
    sys.exit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
    if roop.globals.headless:
        start()
    else:
        window = ui.init(start, destroy)
        window.mainloop()


# ============================================================
# 새로운 함수들: SimSwap, GFPGAN, FOMM, Real-ESRGAN 처리 함수
# ============================================================
def process_video_with_simswap(source_image: str, target_video: str, output_video: str) -> None:
    """
    SimSwap을 사용하여 source_image와 target_video를 기반으로 얼굴 스왑을 진행합니다.
    이 함수는 외부 SimSwap 모듈의 swap_video() 함수를 호출합니다.
    """
    try:
        from simswap.test_video_swapsingle import swap_video
        swap_video(source_img=source_image, target_video=target_video, output_path=output_video)
    except Exception as e:
        print(f"SimSwap 처리 중 오류 발생: {e}")
        import sys
        sys.exit(1)


def enhance_faces_gfpgan(input_video: str, output_video: str) -> None:
    """
    GFPGAN을 사용하여 input_video의 얼굴 디테일을 향상시키고 output_video로 저장합니다.
    """
    try:
        from gfpgan import GFPGANer
        gfpgan = GFPGANer(model_path='gfpgan.pth', upscale=2)
    
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # GFPGAN 처리
            _, restored_frame, _ = gfpgan.enhance(frame, paste_back=True)
            out.write(restored_frame)
    
        cap.release()
        out.release()
    except Exception as e:
        print(f"GFPGAN 처리 중 오류 발생: {e}")
        import sys
        sys.exit(1)


def apply_fomm(source_image: str, input_video: str, output_video: str) -> None:
    """
    FOMM을 사용하여 input_video에 대한 얼굴 움직임 및 표정 보정을 진행하고 output_video로 저장합니다.
    이 함수는 프로젝트 내에 새로 만든 래퍼 함수 first_order_motion을 호출합니다.
    """
    try:
        from roop.fomm_wrapper import first_order_motion
        
        # config_path와 checkpoint_path는 프로젝트 환경에 맞게 지정합니다.
        config_path = "config/vox-256.yaml"          # 예시: 설정 파일 경로
        checkpoint_path = "checkpoints/vox-cpk.pth.tar"  # 예시: 체크포인트 파일 경로
        
        # first_order_motion 함수 호출
        first_order_motion(source_image, input_video, output_video, config_path, checkpoint_path, cpu=False)
    except Exception as e:
        print(f"FOMM 처리 중 오류 발생: {e}")
        import sys
        sys.exit(1)


def enhance_video_with_realesrgan(input_video: str, output_video: str, scale: int = 2, device: str = 'cuda') -> None:
    """
    Real-ESRGAN을 사용하여 input_video를 해상도 업스케일 처리한 후 output_video로 저장합니다.
    """
    try:
        from realesrgan import RealESRGANer
        model = RealESRGANer(device, scale=scale)
        model_path = f'RealESRGAN_x{scale}.pth'
        if not os.path.exists(model_path):
            print(f"{model_path} 파일이 존재하지 않습니다. 모델 파일을 수동으로 다운로드하세요.")
            import sys
            sys.exit(1)
        model.load_weights(model_path)
    
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * scale
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * scale
    
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            sr_frame = model.predict(frame)
            out.write(sr_frame)
    
        cap.release()
        out.release()
    except Exception as e:
        print(f"Real-ESRGAN 처리 중 오류 발생: {e}")
        import sys
        sys.exit(1)
