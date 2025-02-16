#!/usr/bin/env python3

from roop import core

if __name__ == '__main__':
    # roop.core.run() 내부에서
    # 1. SimSwap 기반 얼굴 스왑
    # 2. GFPGAN을 통한 얼굴 디테일 보정
    # 3. FOMM을 통한 얼굴 움직임/표정 보정
    # 4. Real-ESRGAN을 통한 해상도 업스케일
    # 의 후처리 파이프라인을 순차적으로 실행합니다.
    core.run()
