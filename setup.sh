#!/bin/bash

# 시스템 패키지 설치
apt-get update && apt-get install -y ffmpeg portaudio19-dev python3-pyaudio

# ALSA 오류 방지를 위한 디렉토리 생성
mkdir -p /tmp/runtime-root

# Python 패키지 설치
pip install -q -r requirements.txt