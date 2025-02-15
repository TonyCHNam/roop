import cv2
import numpy as np
import os
import glob
import mimetypes
import platform
import shutil
import ssl
import subprocess
import urllib.request
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from realesrgan import RealESRGAN  # pip install realesrgan

import roop.globals

# -------------------------------
# post_processing.py의 enhance_video 함수 (자동 다운로드 기능 추가 및 모델 URL 수정)
# -------------------------------

def download_model(model_path: str, url: str) -> bool:
    """
    주어진 URL에서 모델 파일을 다운로드하여 model_path에 저장합니다.
    """
    try:
        print(f"Downloading {model_path} from {url} ...")
        urllib.request.urlretrieve(url, model_path)
        print("Download completed.")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def enhance_video(input_video: str, output_video: str, scale: int = 2, device: str = 'cuda') -> None:
    """
    Real-ESRGAN을 사용하여 input_video를 super-resolution 처리한 후 output_video로 저장합니다.
    
    :param input_video: 원본 영상 파일 경로
    :param output_video: 후처리된 영상 파일 경로
    :param scale: 업스케일 배율 (일반적으로 2 또는 4)
    :param device: 사용할 디바이스 ('cuda' 또는 'cpu')
    """
    # RealESRGAN 모델 초기화 (scale에 따라 모델 파일이 달라집니다)
    model = RealESRGAN(device, scale=scale)
    
    model_path = f"RealESRGAN_x{scale}.pth"
    # 모델 가중치 파일이 없으면 자동 다운로드 (Colab 환경용)
    if not os.path.exists(model_path):
        # 제공된 URL을 사용하여 모델 파일을 다운로드합니다.
        download_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
        if not download_model(model_path, download_url):
            print(f"Failed to download {model_path}.")
            return
    model.load_weights(model_path)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Unable to open input video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * scale
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * scale

    # 호환성을 위해 mp4 컨테이너와 적절한 코덱 사용 (예: 'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {frame_count}")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Real-ESRGAN으로 프레임 업스케일 (cv2에서 BGR 형식)
        sr_frame = model.predict(frame)  # 모델에 따라 predict 함수 사용
        out.write(sr_frame)
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames")
    
    cap.release()
    out.release()
    print("Super resolution processing completed!")

# -------------------------------
# utilities.py의 기타 함수들
# -------------------------------

TEMP_DIRECTORY = 'temp'
TEMP_VIDEO_FILE = 'temp.mp4'

# macOS에서 SSL 검증 우회를 위한 monkey patch
if platform.system().lower() == 'darwin':
    ssl._create_default_https_context = ssl._create_unverified_context

def run_ffmpeg(args: List[str]) -> bool:
    commands = ['ffmpeg', '-hide_banner', '-loglevel', roop.globals.log_level]
    commands.extend(args)
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except Exception:
        pass
    return False

def detect_fps(target_path: str) -> float:
    command = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        target_path
    ]
    output = subprocess.check_output(command).decode().strip().split('/')
    try:
        numerator, denominator = map(int, output)
        return numerator / denominator
    except Exception:
        pass
    return 30

def extract_frames(target_path: str, fps: float = 30) -> bool:
    temp_directory_path = get_temp_directory_path(target_path)
    temp_frame_quality = roop.globals.temp_frame_quality * 31 // 100
    return run_ffmpeg([
        '-hwaccel', 'auto',
        '-i', target_path,
        '-q:v', str(temp_frame_quality),
        '-pix_fmt', 'rgb24',
        '-vf', 'fps=' + str(fps),
        os.path.join(temp_directory_path, '%04d.' + roop.globals.temp_frame_format)
    ])

def create_video(target_path: str, fps: float = 30) -> bool:
    """
    단일 pass 인코딩을 사용하여 호환성이 높은 MP4 파일을 생성합니다.
    - libx264 인코더를 사용하여 인코딩합니다.
    - -movflags +faststart, -profile:v main, -level:v 3.1 옵션을 추가해 표준 프로파일 및 호환성을 강화합니다.
    - 오디오는 AAC 코덱(-c:a aac -b:a 128k)으로 인코딩합니다.
    """
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    crf_value = 18  # 최상의 화질과 파일 크기 균형 (낮을수록 화질이 좋지만 파일 크기가 커짐)

    commands = [
        '-hwaccel', 'auto',
        '-r', str(fps),
        '-i', os.path.join(temp_directory_path, '%04d.' + roop.globals.temp_frame_format),
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', str(crf_value),
        '-c:a', 'aac',
        '-b:a', '128k',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-profile:v', 'main',
        '-level:v', '3.1',
        '-y', temp_output_path
    ]
    return run_ffmpeg(commands)

def restore_audio(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    done = run_ffmpeg([
        '-i', temp_output_path,
        '-i', target_path,
        '-c:v', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-y', output_path
    ])
    if not done:
        move_temp(target_path, output_path)

def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob(os.path.join(glob.escape(temp_directory_path), '*.' + roop.globals.temp_frame_format))

def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)

def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_VIDEO_FILE)

def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Optional[str]:
    if source_path and target_path and output_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if os.path.isdir(output_path):
            return os.path.join(output_path, source_name + '-' + target_name + target_extension)
    return output_path

def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)

def move_temp(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)

def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not roop.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)

def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(('png', 'jpg', 'jpeg', 'webp'))

def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith('image/'))
    return False

def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith('video/'))
    return False

def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        download_file_path = os.path.join(download_directory_path, os.path.basename(url))
        if not os.path.exists(download_file_path):
            request = urllib.request.urlopen(url)  # type: ignore[attr-defined]
            total = int(request.headers.get('Content-Length', 0))
            with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                urllib.request.urlretrieve(
                    url,
                    download_file_path,
                    reporthook=lambda count, block_size, total_size: progress.update(block_size)
                )  # type: ignore[attr-defined]

def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))
