# WhisperX 기반 자막 생성기

이 프로젝트는 WhisperX를 사용하여 음성 파일에서 자막을 자동으로 생성하는 도구입니다. 특히 Google Colab 환경에서 사용하기 쉽게 설계되었습니다.

## 주요 기능

- 여러 오디오/비디오 파일 (mp3, mp4, webm, wav, m4a, mkv)에서 자막 생성
- YouTube 링크에서 오디오 추출 및 자막 생성
- 다국어 지원 (영어, 한국어 등 다양한 언어)
- 화자 분리 기능 (다중 화자 대화 자막 생성)
- 고품질 단어 단위 타임스탬핑
- 자동 언어 감지
- 긴 오디오 파일 자동 분할 처리
- 한국어 번역 기능

## Google Colab 사용 방법

1. Google Colab에서 노트북을 열고 다음 코드를 실행합니다:

```python
# Google Drive 연동 및 설치
from google.colab import drive
drive.mount('/content/drive')

# 저장소 클론
!git clone https://github.com/limbic-Kim/subtitle5.git
%cd subtitle5

# 환경 설정
%run colab_setup.py

# 실행 예시 (파일 모드)
!python app.py --mode files --input_file_dir "/content/drive/MyDrive/inputs" --model "large-v2" --lang "english" --subformat "SRT" --use_diarization
```

2. 구글 드라이브의 "inputs" 폴더에 처리할 오디오/비디오 파일을 업로드합니다.
3. 실행이 완료되면 구글 드라이브의 "outputs" 폴더에서 생성된 자막 파일을 확인할 수 있습니다.

## 명령어 옵션

```
--mode: files 또는 youtubes (파일 처리 또는 YouTube 링크 처리)
--input_file_dir: 입력 파일이 있는 디렉토리 경로 (mode=files인 경우)
--youtube_links_input: 처리할 YouTube 링크 (mode=youtubes인 경우)
--model: 사용할 Whisper 모델 크기 (tiny, base, small, medium, large-v1, large-v2, large-v3)
--lang: 음성의 언어 (english, korean 등)
--subformat: 자막 형식 (SRT, WebVTT, txt)
--translate: 번역 활성화 (플래그)
--force_skip_translation: 한국어일 경우에도 번역 강제 실행 (플래그)
--use_diarization: 화자 분리 기능 활성화 (플래그)
```

## 화자 분리 기능 (WhisperX)

화자 분리 기능을 사용하면 여러 사람이 대화하는 오디오에서 각 화자를 구분하여 자막을 생성합니다. 
이 기능을 활성화하려면 `--use_diarization` 플래그를 사용하세요:

```
!python app.py --mode files --input_file_dir "/content/drive/MyDrive/inputs" --model "large-v2" --lang "english" --subformat "SRT" --use_diarization
```

생성된 자막에는 각 문장 앞에 화자 식별 정보(예: [SPEAKER_00], [SPEAKER_01])가 추가됩니다.

## 요구 사항

이 프로젝트는 다음 라이브러리를 사용합니다:
- whisperx (고급 음성 인식 및 화자 분리)
- torch, torchaudio (GPU 가속 처리)
- moviepy (미디어 파일 처리)
- pyannote.audio (화자 분리)
- 기타 종속성 (requirements.txt 참조)

## 주의사항

- 화자 분리 기능은 추가적인 계산 리소스를 필요로 합니다. Google Colab의 무료 버전에서는 메모리 한계에 도달할 수 있습니다.
- 긴 오디오 파일의 경우 자동으로 분할되어 처리되며, 이는 처리 시간을 증가시킬 수 있습니다.
- 모델 크기에 따라 정확도와 처리 속도가 달라집니다. 더 큰 모델은 더 정확하지만 처리 시간이 더 오래 걸립니다. 