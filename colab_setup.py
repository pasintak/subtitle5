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
        # 먼저 CUDA 버전 확인
        print("Checking CUDA version...")
        os.system("nvcc --version")
        
        # NumPy 설치 - 다른 버전 시도
        print("Installing numpy...")
        os.system("pip install numpy==1.24.3")  # 변경된 버전
        
        # SciPy 설치 - 호환되는 버전
        print("Installing scipy...")
        os.system("pip install scipy==1.10.1")  # 변경된 버전
        
        # CPU 전용 PyTorch 설치 (CUDA 문제 방지)
        print("Installing PyTorch (CPU only)...")
        os.system("pip install torch==2.0.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cpu")
        
        # 기타 기본 패키지 설치
        print("Installing basic dependencies...")
        os.system("pip install setuptools-rust==1.5.2 transformers==4.30.0 tokenizers==0.13.3")
        
        # 의존성 설치
        print("Installing Whisper...")
        os.system("pip install openai-whisper==20230918")
        
        print("Installing other dependencies...")
        os.system("pip install moviepy pyyaml pytube gradio==3.35.0")
        
        # 오디오 처리 라이브러리 설치
        print("Installing audio processing libraries...")
        os.system("pip install librosa==0.10.1 soundfile")
        
        # 추가 의존성
        print("Installing additional dependencies...")
        os.system("pip install pandas==2.0.2 scikit-learn==1.3.0")
        
        # faster-whisper 설치
        print("Installing faster-whisper...")
        os.system("pip install faster-whisper==0.6.0")
        
        # pyannote.audio 설치
        print("Installing pyannote.audio...")
        os.system("pip install pyannote.audio==2.1.1")
        
        # 설치 간 간격 두기 (충돌 방지)
        time.sleep(2)
        
        # WhisperX 직접 코드 복사 방식으로 설치
        print("Installing WhisperX via direct source copy...")
        # Clone the repository
        os.system("rm -rf whisperx_temp")  # 기존 디렉토리 제거
        os.system("git clone https://github.com/m-bain/whisperX.git whisperx_temp")
        
        # 필요한 디렉토리 생성
        os.system("mkdir -p whisperx")
        
        # 소스 코드 복사
        os.system("cp -r whisperx_temp/whisperx/* whisperx/")
        os.system("cp whisperx_temp/setup.py .")
        
        # PYTHONPATH에 현재 디렉토리 추가
        current_dir = os.getcwd()
        if 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] = f"{current_dir}:{os.environ['PYTHONPATH']}"
        else:
            os.environ['PYTHONPATH'] = current_dir
        
        # sys.path에 현재 디렉토리 추가
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        print(f"Added {current_dir} to PYTHONPATH and sys.path")
        
        # 설치 확인을 위한 대기
        time.sleep(3)
        
        # 설치된 패키지 확인
        print("Checking installed packages...")
        os.system("pip list | grep numpy")
        os.system("pip list | grep torch")
        
        # Python 경로 출력
        print(f"Python path: {sys.path}")
        
        # 디렉토리 내용 확인
        print("Checking directory contents...")
        os.system("ls -la")
        os.system("ls -la whisperx")
        
        try:
            import numpy as np
            import torch
            
            print(f"NumPy version: {np.__version__}")
            print(f"PyTorch version: {torch.__version__}")
            
            # whisperx 임포트 시도
            print("Trying to import whisperx...")
            import whisperx
            
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
            print("whisperx 디렉토리 내용:")
            os.system("ls -la whisperx")
            
            # __init__.py 파일 존재 확인
            if not os.path.exists("whisperx/__init__.py"):
                print("Creating __init__.py file...")
                with open("whisperx/__init__.py", "w") as f:
                    f.write("# WhisperX initialization file\n")
            
            # 마지막 시도: pip로 설치
            print("\n최종 시도: pip로 일반 설치")
            os.system("pip install git+https://github.com/m-bain/whisperX.git")
            
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