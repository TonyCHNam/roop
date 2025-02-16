ROOP

ROOP는 얼굴 스왑 및 비디오 편집을 위한 GAN 기반 도구입니다.
이 프로젝트는 SimSwap, GFPGAN, FOMM, Real-ESRGAN 등을 조합하여 고품질의 얼굴 스왑 비디오를 생성합니다.


주요 기능
SimSwap: 얼굴 스왑을 수행합니다.
GFPGAN: 얼굴의 디테일과 품질을 보정합니다.
FOMM (First Order Motion Model): 얼굴의 움직임 및 표정을 자연스럽게 보정합니다.
Real-ESRGAN: 비디오 해상도를 업스케일합니다.



요구 사항
Python 3.9 이상
ffmpeg (시스템에 맞게 설치)
기타 Python 패키지: requirements.txt 및 requirements-headless.txt 파일을 참고하세요.



설치 방법

1. Python 패키지 설치

먼저, 아래 명령어를 통해 필요한 Python 패키지를 설치하세요:

pip install -r requirements.txt
pip install -r requirements-headless.txt


주의:
requirements.txt 및 requirements-headless.txt 파일에서는 simswap과 first_order_model 항목이 제거되어 있습니다. 이 두 모듈은 pip로 설치할 수 없으므로, 아래 별도의 클론 지침을 따르세요.


2. SimSwap 저장소 클론

ROOP는 얼굴 스왑 처리를 위해 SimSwap 저장소를 사용합니다.
프로젝트 루트 디렉토리에서 아래 명령어를 실행하여 SimSwap 저장소를 클론하세요:

git clone https://github.com/neuralchen/SimSwap.git SimSwap

이렇게 클론된 SimSwap 폴더는 ROOP에서 대문자 "SimSwap" 모듈 경로로 사용됩니다.


3. First Order Motion Model (FOMM) 저장소 클론

FOMM은 pip 설치가 불가능하므로, 프로젝트 루트 디렉토리에서 아래 명령어를 실행하여 클론하세요:

git clone https://github.com/AliaksandrSiarohin/first-order-model.git first_order_model


클론 후, ROOP에서는 roop/fomm_wrapper.py를 통해 FOMM 기능을 호출합니다.
(필요 시 PYTHONPATH에 first_order_model 경로를 추가하거나, 프로젝트 루트에 위치하는지 확인하세요.)



실행 방법
ROOP를 실행하려면, 다음 명령어를 사용하세요:


python run.py --source path/to/source.jpg --target path/to/target.mp4 --output path/to/output.mp4


자세한 옵션은 다음 명령어로 확인할 수 있습니다:

python run.py -h


GitHub Actions CI
CI에서는 simswap 및 first_order_model 패키지는 pip로 설치하지 않습니다.
대신, CI 워크플로우(.github/workflows/ci.yml)에서는 아래와 같이 SimSwap 저장소를 클론하고, PYTHONPATH에 추가하여 모듈 경로 문제를 해결합니다.

- name: Clone SimSwap repository
  run: git clone https://github.com/neuralchen/SimSwap.git SimSwap

- name: Set PYTHONPATH for SimSwap
  run: echo "PYTHONPATH=${{ github.workspace }}/SimSwap:$PYTHONPATH" >> $GITHUB_ENV


이와 같이 설정하면, ROOP는 클론된 SimSwap 저장소 내의 모듈(예: SimSwap.test_video_swapsingle)을 올바르게 사용할 수 있습니다.

라이선스
이 프로젝트는 MIT 라이선스에 따릅니다.
