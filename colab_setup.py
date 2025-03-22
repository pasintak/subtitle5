import os
import sys
import subprocess
import time


def setup_environment():
    os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"

    # 시스템 패키지 설치
    print("Installing system packages...")
    os.system(
        "apt-get update && apt-get install -y ffmpeg portaudio19-dev python3-pyaudio"
    )
    os.system("mkdir -p /tmp/runtime-root")

    # pip 업그레이드
    print("Upgrading pip...")
    os.system("pip install --upgrade pip setuptools wheel")
    
    # 완전 정리 (강제 실행)
    print("Completely cleaning the environment...")
    packages_to_uninstall = [
        "numpy", "torch", "torchaudio", "whisper", "whisperx", 
        "pyannote.audio", "transformers", "tokenizers", "faster-whisper",
        "scipy", "numba", "librosa", "pandas", "scikit-learn"
    ]
    
    for package in packages_to_uninstall:
        try:
            subprocess.run(["pip", "uninstall", "-y", package], stdout=subprocess.PIPE)
        except:
            pass
    
    # pip 캐시 정리
    print("Cleaning pip cache...")
    os.system("pip cache purge")
    
    try:
        # NumPy 설치 - 안정적인 버전
        print("Installing numpy...")
        os.system("pip install numpy==1.22.4")
        
        # SciPy 설치 - NumPy와 호환되는 버전
        print("Installing scipy...")
        os.system("pip install scipy==1.8.1")
        
        # 기본 패키지 설치 (특정 버전으로 고정)
        print("Installing PyTorch...")
        os.system("pip install torch==1.13.1 torchaudio==0.13.1")
        
        # 기타 기본 패키지 설치
        print("Installing basic dependencies...")
        os.system("pip install setuptools-rust==1.5.2 transformers==4.27.4 tokenizers==0.13.2")
        
        # 의존성 설치
        print("Installing Whisper...")
        os.system("pip install openai-whisper==20230314")
        
        print("Installing other dependencies...")
        os.system("pip install moviepy pyyaml pytube gradio==3.32.0")
        
        # 오디오 처리 라이브러리 설치
        print("Installing audio processing libraries...")
        os.system("pip install librosa==0.9.2 soundfile")
        
        # 추가 의존성
        print("Installing additional dependencies...")
        os.system("pip install pandas==1.5.3 scikit-learn==1.2.1")
        
        # faster-whisper 설치
        print("Installing faster-whisper...")
        os.system("pip install faster-whisper==0.4.1")
        
        # pyannote.audio 설치
        print("Installing pyannote.audio...")
        os.system("pip install pyannote.audio==2.1.1")
        
        # 설치 간 간격 두기 (충돌 방지)
        time.sleep(2)
        
        # WhisperX 설치 방법 수정
        print("Installing WhisperX...")
        # 1. 기존 설치 확실히 제거
        os.system("pip uninstall -y whisperx")
        
        # 2. 저장소 직접 클론
        os.system("git clone https://github.com/m-bain/whisperX.git ./whisperx_temp")
        os.chdir("./whisperx_temp")
        
        # 3. 직접 설치
        os.system("pip install -e .")
        os.chdir("..")
        
        # 설치 확인을 위한 대기
        time.sleep(5)
        
        # 패키지 설치 목록 확인
        print("Checking installed packages...")
        os.system("pip list | grep whisperx")
        os.system("pip list | grep numpy")
        os.system("pip list | grep torch")
        
        # Python 경로 출력
        print(f"Python path: {sys.path}")
        
        try:
            import numpy as np
            import torch
            import whisperx
            
            print(f"NumPy version: {np.__version__}")
            print(f"PyTorch version: {torch.__version__}")
            print("WhisperX 설치 완료")
            
            # WhisperX 기본 기능 확인
            print("Testing WhisperX functionality...")
            has_load_model = hasattr(whisperx, 'load_model')
            has_align = hasattr(whisperx, 'align')
            
            if has_load_model and has_align:
                print("✅ WhisperX 기능 확인 완료")
            else:
                print("⚠️ WhisperX 일부 기능이 누락되었습니다.")
            
            return True
        except ImportError as e:
            print(f"Error importing packages: {e}")
            
            # 추가 디버깅 정보
            print("\n디버깅 정보:")
            print("1. whisperx 패키지 위치 확인:")
            os.system("find /usr -name whisperx")
            os.system("find /content -name whisperx")
            
            print("\n2. PYTHONPATH 확인:")
            print(os.environ.get('PYTHONPATH', '설정되지 않음'))
            
            # 마지막 시도: 직접 경로 추가
            print("\n3. whisperx 소스 직접 복사:")
            os.system("cp -r ./whisperx_temp/whisperx .")
            sys.path.append(os.getcwd())
            
            try:
                import whisperx
                print("마지막 방법으로 whisperx 임포트 성공!")
                return True
            except ImportError as e2:
                print(f"최종 임포트 실패: {e2}")
                return False
    except Exception as e:
        print(f"Setup error: {e}")
        return False


if __name__ == "__main__":
    setup_environment()