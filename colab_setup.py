import os
import sys


def setup_environment():
    os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"

    os.system(
        "apt-get update && apt-get install -y ffmpeg portaudio19-dev python3-pyaudio"
    )
    os.system("mkdir -p /tmp/runtime-root")

    # WhisperX와 필요한 의존성 설치
    os.system(
        "pip install -q openai-whisper faster-whisper transformers gradio==3.37.0 pytube moviepy pyyaml torch torchaudio pyannote.audio setuptools-rust"
    )
    os.system(
        "pip install -q git+https://github.com/m-bain/whisperX.git"
    )

    # 설치 확인
    try:
        import whisperx

        print("WhisperX 설치 완료")
        return True
    except ImportError as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    setup_environment()
