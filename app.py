# Standard library imports
import argparse
import itertools
import math
import os, re
import shutil
import time
import traceback
import unicodedata
import urllib
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Union, Tuple

try:
    # Third party imports
    # import gradio as gr
    import numpy as np
    import requests
    import torch
    import yaml
    from moviepy.editor import VideoFileClip

    # Local application imports
    # from .base_interface import BaseInterface
    import whisperx  # Whisper 대신 WhisperX로 수정
    import gc  # 메모리 관리를 위한 가비지 컬렉션 import
    from typing import List
    
    # 라이브러리 버전 출력
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    try:
        print(f"WhisperX version: {whisperx.__version__}")
    except AttributeError:
        print("WhisperX version information not available")
    
    # 모듈 구조와 버전 호환성 체크
    WHISPERX_COMPATIBLE = True
    
except ImportError as e:
    print(f"Import Error: {e}")
    print("필요한 라이브러리가 설치되지 않았습니다. colab_setup.py를 실행해 주세요.")
    WHISPERX_COMPATIBLE = False
    raise SystemExit(1)
except Exception as e:
    print(f"오류 발생: {e}")
    print("라이브러리 로드 중 문제가 발생했습니다. colab_setup.py를 다시 실행해 주세요.")
    WHISPERX_COMPATIBLE = False
    raise SystemExit(1)

DEFAULT_MODEL_SIZE = "large-v2"

# 모듈 구조와 버전 체크
def check_whisperx_compatibility():
    """WhisperX의 호환성을 확인합니다."""
    try:
        # 기본 기능 테스트
        has_load_model = hasattr(whisperx, 'load_model')
        has_diarization = hasattr(whisperx, 'DiarizationPipeline')
        has_align = hasattr(whisperx, 'align')
        
        if not (has_load_model and has_diarization and has_align):
            missing = []
            if not has_load_model: missing.append('load_model')
            if not has_diarization: missing.append('DiarizationPipeline')
            if not has_align: missing.append('align')
            print(f"WhisperX에 필요한 기능이 누락되었습니다: {', '.join(missing)}")
            return False
            
        print("WhisperX 호환성 확인 완료")
        return True
    except Exception as e:
        print(f"WhisperX 호환성 확인 중 오류 발생: {e}")
        return False

# 호환성 체크 실행
whisperx_compatible = check_whisperx_compatibility()
if not whisperx_compatible:
    print("WhisperX가 올바르게 설치되지 않았습니다. colab_setup.py를 다시 실행해주세요.")


class BaseInterface:
    def __init__(self):
        pass

    @staticmethod
    def release_cuda_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()

    @staticmethod
    def remove_input_files(file_paths: List[str]):
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
            os.remove(file_path)


class WhisperInference(BaseInterface):
    def __init__(self):
        """
        WhisperInference 클래스 초기화
        """
        super().__init__()
        self.model = None
        self.diarize_model = None
        self.current_model_size = None
        self.current_compute_type = None
        self.use_diarization = False  # 디폴트는 화자 분리 사용 안 함
        
        # 시스템 리소스 체크 및 설정
        if torch.cuda.is_available():
            self.device = "cuda"
            # CUDA 메모리 정보 출력
            try:
                print(f"CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
                print(f"CUDA 사용 가능 메모리: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
            except Exception as e:
                print(f"CUDA 정보 확인 중 오류: {e}")
        else:
            self.device = "cpu"
            print("CUDA 사용 불가. CPU 모드로 실행합니다.")

    @staticmethod
    def convert_video_to_audio(input_video_path, output_audio_path, bitrate="64k"):
        from moviepy.editor import VideoFileClip
        video = VideoFileClip(input_video_path)
        audio = video.audio
        audio.write_audiofile(output_audio_path, bitrate=bitrate)

    @staticmethod
    def split_large_audio(input_audio_path, temp_folder, max_duration=7200, max_size=100 * 1024 * 1024):
        from moviepy.editor import VideoFileClip
        import math
        import shutil

        # Ensure the temp folder exists
        os.makedirs(temp_folder, exist_ok=True)
        print(f"Ensured the temp folder {temp_folder} exists")

        # Check the audio file duration and size
        clip = VideoFileClip(str(input_audio_path))
        duration = clip.duration
        file_size = os.path.getsize(input_audio_path)
        print(f"Checked the audio file. Duration: {duration} seconds, Size: {file_size} bytes")

        # Check if splitting is necessary
        if duration > max_duration or file_size > max_size:
            num_chunks = math.ceil(duration / max_duration)
            audio_chunks = []
            print(f"Splitting necessary. Will split into {num_chunks} chunks")

            for i in range(num_chunks):
                start = i * max_duration
                end = start + max_duration if (i + 1) < num_chunks else duration
                chunk_filename = f"chunk_{i}.mp3"
                chunk_path = os.path.join(temp_folder, chunk_filename)
                WhisperInference.convert_video_to_audio(input_video_path=input_audio_path, output_audio_path=chunk_path)
                audio_chunks.append(chunk_path)
                print(f"Created chunk {i} from {start} to {end} seconds")

            return audio_chunks
        else:
            # Return the input audio path in a list if no splitting is needed
            print("No splitting needed")
            return [input_audio_path]

    def adjust_srt_timestamps(self, parsed_subtitles, last_timestamp, last_index=0):
        adjusted_subtitles = []  # New list to store updated subtitles
        for i, subtitle in enumerate(parsed_subtitles, start=1):  # Added start=1 to start indexing from 1
            start, end = subtitle['timestamp'].split(' --> ')

            # Handle milliseconds
            start_h, start_m, start_s_ms = start.split(':')
            start_s, start_ms = start_s_ms.split(',')
            end_h, end_m, end_s_ms = end.split(':')
            end_s, end_ms = end_s_ms.split(',')

            # Convert to seconds
            start_time = float(start_h) * 3600 + float(start_m) * 60 + float(start_s) + float(start_ms) / 1000
            end_time = float(end_h) * 3600 + float(end_m) * 60 + float(end_s) + float(end_ms) / 1000

            # Adjust times
            adjusted_start_time = start_time + last_timestamp
            adjusted_end_time = end_time + last_timestamp

            # Convert back to timestamp format
            adjusted_start = f'{int(adjusted_start_time // 3600):02}:{int((adjusted_start_time % 3600) // 60):02}:{int(adjusted_start_time % 60):02},{int((adjusted_start_time % 60 % 1) * 1000):03}'
            adjusted_end = f'{int(adjusted_end_time // 3600):02}:{int((adjusted_end_time % 3600) // 60):02}:{int(adjusted_end_time % 60):02},{int((adjusted_end_time % 60 % 1) * 1000):03}'

            # Update subtitle
            subtitle['timestamp'] = f'{adjusted_start} --> {adjusted_end}'
            subtitle['index'] = int(last_index) + i  # Update subtitle index

            adjusted_subtitles.append(subtitle)  # Append updated subtitle to the new list

        return adjusted_subtitles  # Return the new list

    def transcribe_file(self,
                        fileobjs: list,
                        model_size: str,
                        lang: str,
                        file_format: str,
                        istranslate: bool,
                        add_timestamp: bool,
                        beam_size: int,
                        log_prob_threshold: float,
                        no_speech_threshold: float,
                        compute_type: str,
                        # progress=gr.progress()
                        ):
        print(f'model_size = {model_size}')

        with open('config.yaml') as file:
            config = yaml.safe_load(file)

        my_drive = config['my_drive']

        try:
            # WhisperX에 맞게 업데이트된 모델 로드 방식
            self.update_model_if_needed(model_size=model_size, compute_type=compute_type)

            files_info = {}
            for fileobj in fileobjs:
                # WhisperX용 오디오 로드
                audio = whisperx.load_audio(fileobj)

                # WhisperX로 음성 인식 수행
                result, elapsed_time = self.transcribe(
                    audio=audio,
                    lang=lang,
                    istranslate=istranslate,
                    beam_size=beam_size,
                    log_prob_threshold=log_prob_threshold,
                    no_speech_threshold=no_speech_threshold,
                    compute_type=compute_type,
                )

                file_name, file_ext = os.path.splitext(os.path.basename(fileobj))

                if lang == 'Korean' or lang == 'ko':
                    file_name = '(ko)' + file_name

                try:
                    subtitle = self.generate_and_write_file(
                        file_name=file_name,
                        transcribed_segments=result,
                        add_timestamp=add_timestamp,
                        file_format=file_format
                    )

                except Exception as e:
                    print(f"Error generate_and_write_file: {str(e)}")
                    m = traceback.format_exc()
                    print(m)

                    file_name = safe_filename(file_name)
                    print(f"Renaming file name... {file_name} ")
                    subtitle = self.generate_and_write_file(
                        file_name=file_name,
                        transcribed_segments=result,
                        add_timestamp=True,
                        file_format=file_format
                    )

                files_info[file_name] = {"subtitle": subtitle, "elapsed_time": elapsed_time}
                if 'chunk_64_' in file_name:
                    try:
                        if add_timestamp:
                            output_path = os.path.join("outputs", f"{file_name}-{timestamp}")
                        else:
                            output_path = os.path.join("outputs", f"{file_name}")

                        if file_format == "SRT":
                            os.remove(f"{my_drive}/{output_path}.srt")
                    except Exception as e:
                        print(f"Error generate_and_write_file: {str(e)}")
                        m = traceback.format_exc()
                        print(m)

            total_result = ''
            total_time = 0
            for file_name, info in files_info.items():
                total_result += '------------------------------------\n'
                total_result += f'{file_name}\n\n'
                total_result += f"{info['subtitle']}"
                total_time += info["elapsed_time"]

            final_msg1 = f"Done in {self.format_time(total_time)}! Subtitle is in the outputs folder.\n\n{total_result}"
            final_msg2 = f"Done in {self.format_time(total_time)}!"
            print(final_msg1)
            return final_msg1

        except Exception as e:
            print(f"Error transcribing file: {str(e)}")
            m = traceback.format_exc()
            print(m)
            return f"Error transcribing file: {str(e)}"
        finally:
            self.release_cuda_memory()
            # 메모리 정리
            gc.collect()
            # 파일 삭제
            self.remove_input_files(fileobjs)

    def transcribe_youtube(self,
                           youtubelink: str,
                           model_size: str,
                           lang: str,
                           file_format: str,
                           istranslate: bool,
                           add_timestamp: bool,
                           beam_size: int,
                           log_prob_threshold: float,
                           no_speech_threshold: float,
                           compute_type: str,
                           # progress=gr.Progress(),
                           ):

        try:
            self.update_model_if_needed(model_size=model_size, compute_type=compute_type)

            # 유튜브에서 오디오 로드
            yt = get_ytdata(youtubelink)
            audio = whisperx.load_audio(get_ytaudio(yt))

            # WhisperX로 변환
            result, elapsed_time = self.transcribe(
                audio=audio,
                lang=lang,
                istranslate=istranslate,
                beam_size=beam_size,
                log_prob_threshold=log_prob_threshold,
                no_speech_threshold=no_speech_threshold,
                compute_type=compute_type,
            )

            file_name = safe_filename(yt.title)
            subtitle = self.generate_and_write_file(
                file_name=file_name,
                transcribed_segments=result,
                add_timestamp=add_timestamp,
                file_format=file_format
            )

            return f"Done in {self.format_time(elapsed_time)}! Subtitle file is in the outputs folder.\n\n{subtitle}"
        except Exception as e:
            print(f"Error transcribing youtube video: {str(e)}")
            m = traceback.format_exc()
            print(m)
            return f"Error transcribing youtube video: {str(e)}"
        finally:
            try:
                if 'yt' not in locals():
                    yt = get_ytdata(youtubelink)
                    file_path = get_ytaudio(yt)
                else:
                    file_path = get_ytaudio(yt)

                self.release_cuda_memory()
                gc.collect()
                self.remove_input_files([file_path])
            except Exception as cleanup_error:
                pass

    def transcribe_mic(self,
                       micaudio: str,
                       model_size: str,
                       lang: str,
                       file_format: str,
                       istranslate: bool,
                       beam_size: int,
                       log_prob_threshold: float,
                       no_speech_threshold: float,
                       compute_type: str,
                       # progress=gr.Progress()
                       ):

        try:
            self.update_model_if_needed(model_size=model_size, compute_type=compute_type)

            # WhisperX로 변환
            result, elapsed_time = self.transcribe(
                audio=micaudio,
                lang=lang,
                istranslate=istranslate,
                beam_size=beam_size,
                log_prob_threshold=log_prob_threshold,
                no_speech_threshold=no_speech_threshold,
                compute_type=compute_type,
            )

            subtitle = self.generate_and_write_file(
                file_name="Mic",
                transcribed_segments=result,
                add_timestamp=True,
                file_format=file_format
            )

            return f"Done in {self.format_time(elapsed_time)}! Subtitle file is in the outputs folder.\n\n{subtitle}"
        except Exception as e:
            print(f"Error transcribing mic: {str(e)}")
            return f"Error transcribing mic: {str(e)}"
        finally:
            self.release_cuda_memory()
            gc.collect()
            self.remove_input_files([micaudio])

    def transcribe(self,
                   audio: Union[str, np.ndarray, torch.Tensor],
                   lang: str,
                   istranslate: bool,
                   beam_size: int,
                   log_prob_threshold: float,
                   no_speech_threshold: float,
                   compute_type: str,
                   # progress: gr.Progress
                   ) -> Tuple[list[dict], float]:

        start_time = time.time()
        print("Starting transcription...")

        if lang == "Automatic Detection":
            lang = None
        elif lang == "English" or lang == "english":
            lang = "en"
        elif lang == "Korean" or lang == "korean":
            lang = "ko"
        
        print(f'Language setting: {lang}')

        try:
            # 오디오가 문자열(파일 경로)인 경우 로드
            if isinstance(audio, str):
                try:
                    audio = whisperx.load_audio(audio)
                    print(f"Audio loaded from file: shape={audio.shape}, min={audio.min()}, max={audio.max()}")
                except Exception as e:
                    print(f"Error loading audio from file: {e}")
                    import traceback
                    traceback.print_exc()
                    raise Exception(f"오디오 파일 로드 실패: {e}")
            
            # WhisperX 트랜스크립션 설정
            if compute_type == "float16":
                compute_type_param = "float16"
            else:
                compute_type_param = "float32"
                
            print(f"Running transcription with model: {self.current_model_size}, compute_type: {compute_type_param}")
            
            # 트랜스크립션 실행
            try:
                # 여러 API 호출 방식 시도
                try:
                    # 기본 API 호출 시도
                    result = self.model.transcribe(
                        audio=audio,
                        language=lang,
                        batch_size=8,
                        beam_size=beam_size,
                        word_timestamps=True
                    )
                    print(f"Transcription completed: {len(result.get('segments', []))} segments")
                except (TypeError, AttributeError, ValueError) as api_error:
                    print(f"기본 API 호출 실패: {api_error}, 대체 방식 시도...")
                    # 대체 API 호출 시도
                    result = self.model.transcribe(
                        audio,
                        language=lang,
                        beam_size=beam_size
                    )
                    print(f"대체 API로 트랜스크립션 완료: {len(result.get('segments', []))} segments")
                
                # 언어 감지 결과
                detected_language = result.get("language", "en")
                print(f"Detected language: {detected_language}")
                
                # 단어 정렬 수행
                try:
                    print("단어 정렬 실행 중...")
                    align_model, align_metadata = whisperx.load_align_model(
                        language_code=detected_language,
                        device=self.device
                    )
                    
                    # 정렬 실행
                    result = whisperx.align(
                        result["segments"], 
                        align_model, 
                        align_metadata, 
                        audio, 
                        self.device,
                        return_char_alignments=False
                    )
                    print("단어 정렬 완료")
                    
                    # 화자 분리 기능 (사용 설정된 경우)
                    if self.use_diarization:
                        try:
                            print("화자 분리 실행 중...")
                            if self.diarize_model is None:
                                print("화자 분리 모델 초기화...")
                                self.diarize_model = whisperx.DiarizationPipeline(
                                    device=self.device
                                )
                            
                            # 화자 분리 수행
                            diarize_segments = self.diarize_model(audio)
                            
                            # 화자 정보 할당
                            try:
                                result = whisperx.assign_word_speakers(diarize_segments, result)
                                speakers = set([s.get('speaker', '') for s in result['segments'] if 'speaker' in s])
                                print(f"화자 분리 완료: {len(speakers)}명의 화자 감지됨")
                            except Exception as e:
                                print(f"화자 정보 할당 중 오류: {str(e)}")
                                import traceback
                                traceback.print_exc()
                        except Exception as e:
                            print(f"화자 분리 중 오류: {str(e)}")
                            import traceback
                            traceback.print_exc()
                    
                except Exception as e:
                    print(f"단어 정렬 중 오류: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # 정렬 실패시 원본 세그먼트 사용
                    if "segments" not in result:
                        print("세그먼트 정보 없음, 빈 리스트 생성")
                        result["segments"] = []
                
                # 결과 세그먼트 변환
                segments_result = []
                for segment in result.get("segments", []):
                    segment_data = {
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment.get("text", ""),
                    }
                    
                    # 화자 정보 추가 (있는 경우)
                    if self.use_diarization and "speaker" in segment:
                        segment_data["speaker"] = segment["speaker"]
                    
                    segments_result.append(segment_data)
                
                elapsed_time = time.time() - start_time
                
                if not segments_result:
                    print("경고: 트랜스크립션에서 세그먼트가 생성되지 않았습니다")
                else:
                    print(f"{len(segments_result)}개의 세그먼트 처리 완료 (소요 시간: {elapsed_time:.2f}초)")
                    
                return segments_result, elapsed_time
                
            except Exception as e:
                print(f"트랜스크립션 오류: {str(e)}")
                import traceback
                traceback.print_exc()
                raise Exception(f"음성 인식 중 오류가 발생했습니다: {str(e)}")
                
        except Exception as e:
            print(f"전체 트랜스크립션 처리 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            # 오류 발생 시 빈 결과 반환
            return [], time.time() - start_time

    def update_model_if_needed(self,
                               model_size: str,
                               compute_type: str,
                               # progress: gr.Progress,
                               ):
        """
        Initialize model if it doesn't match with current model setting
        """
        if compute_type != self.current_compute_type:
            self.current_compute_type = compute_type
        if model_size != self.current_model_size or self.model is None:
            # 기존 모델 정리
            if self.model is not None:
                del self.model
                gc.collect()
                torch.cuda.empty_cache()
            
            self.current_model_size = model_size
            try:
                print(f"Loading WhisperX model: {model_size} with compute_type: {compute_type}")
                
                # WhisperX 모델 로드 다양한 API 버전을 시도하기
                for load_attempt in range(3):
                    try:
                        if load_attempt == 0:
                            # 1. 기본 위치 인자 방식 (최신 버전)
                            self.model = whisperx.load_model(
                                model_size,  # 모델 크기
                                self.device,  # 장치
                                compute_type=compute_type
                            )
                            print("모델 로드 성공 (위치 인자 방식)")
                            break
                        elif load_attempt == 1:
                            # 2. 키워드 인자 방식 (일부 버전)
                            self.model = whisperx.load_model(
                                model=model_size,
                                device=self.device,
                                compute_type=compute_type
                            )
                            print("모델 로드 성공 (model 키워드 인자 방식)")
                            break
                        else:
                            # 3. name 키워드 방식 (이전 버전)
                            self.model = whisperx.load_model(
                                name=model_size,
                                device=self.device,
                                compute_type=compute_type
                            )
                            print("모델 로드 성공 (name 키워드 인자 방식)")
                            break
                    except (TypeError, ValueError, AttributeError) as e:
                        print(f"모델 로드 시도 {load_attempt+1} 실패: {e}")
                        if load_attempt == 2:
                            raise Exception("모든 모델 로드 방식이 실패했습니다.")
                
                # 모델 로드 성공 정보 출력
                print(f"모델 {model_size} 로드 완료 ({self.device} 사용)")
                
                # 메모리 정보 출력 (CUDA 사용 시)
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                    print(f"GPU 메모리 사용량: {allocated:.2f} GB")
                
            except Exception as e:
                print(f"모델 로드 오류: {str(e)}")
                import traceback
                traceback.print_exc()
                print("모델 로드에 실패했습니다. colab_setup.py를 다시 실행해보세요.")
                raise

    def generate_srt_string(self, subtitles):
        """Takes a list of subtitle dictionaries and turns it into an SRT formatted string."""
        srt_entries = []
        for subtitle in subtitles:
            text = subtitle['sentence']
            
            # 자막 딕셔너리에 speaker 키가 있고, self.use_diarization이 활성화된 경우에만 화자 표시
            if self.use_diarization and 'speaker' in subtitle and subtitle['speaker']:
                text = f"[{subtitle['speaker']}] {text}"
            
            srt_entries.append(f"{subtitle['index']}\n{subtitle['timestamp']}\n{text}\n")
        return "\n".join(srt_entries).strip()

    @staticmethod
    def generate_and_write_file(file_name: str,
                                transcribed_segments: list,
                                add_timestamp: bool,
                                file_format: str,
                                ) -> str:
        """
        This method writes subtitle file and returns str to gr.Textbox
        """
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        if add_timestamp:
            output_path = os.path.join("outputs", f"{file_name}-{timestamp}")
        else:
            output_path = os.path.join("outputs", f"{file_name}")
        content = ""

        with open('config.yaml') as file:
            config = yaml.safe_load(file)

        my_drive = config['my_drive']

        if file_format == "SRT":
            content = get_srt(transcribed_segments)
            write_file(content, f"{output_path}.srt")
            try:
                write_file(content, f"{my_drive}/{output_path}.srt")
            except Exception as e:
                print(f"Error transcribing file: {str(e)}")
                m = traceback.format_exc()
                print(m)
                print('구글드라이브 백업 실패. 해당 경로를 확인하세요. ')

        elif file_format == "WebVTT":
            content = get_vtt(transcribed_segments)
            write_file(content, f"{output_path}.vtt")

        elif file_format == "txt":
            content = get_txt(transcribed_segments)
            write_file(content, f"{output_path}.vtt")
        return content

    @staticmethod
    def check_and_split_audio(file_path: str, temp_folder: str, max_duration: float = 7200.0,  # 60.0, # 7200.0, #todo
                              max_file_size: float = 104857600.0) -> list:  # 104857600.0
        from moviepy.editor import VideoFileClip, AudioFileClip

        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder, exist_ok=True)
            print(f"Ensured the temp folder {temp_folder} exists")

        with open('config.yaml') as file:
            config = yaml.safe_load(file)
        max_duration = config['max_duration']
        max_file_size = config['max_file_size']

        chunks = []
        file_size = os.path.getsize(file_path)
        print(f"Checked the file. Size: {file_size} bytes")

        # Check the file extension to determine whether it's an audio or video file
        file_extension = os.path.splitext(file_path)[-1].lower()
        video_extensions = ['.mp4', '.webm', '.mkv']  # mkv 추가
        audio_extensions = ['.mp3', '.wav', '.m4a']
        clip = None
        duration = 0

        if file_extension in video_extensions:
            try:
                clip = VideoFileClip(str(file_path))  # Convert file_path to string
                duration = clip.duration
                print(f"Checked the video file. Duration: {duration} seconds")

                if len(clip.get_frame(0).shape) < 3:
                    clip = AudioFileClip(str(file_path))  # Treat as audio file
                    print("No video frames found. Treating as audio file...")
            except Exception as e:
                print(f"Error processing video file {file_path}: {str(e)}")
                if 'video_fps' in str(e):
                    try:
                        clip = AudioFileClip(str(file_path))  # Convert file_path to string
                        duration = clip.duration
                        print(f"Checked the audio file. Duration: {duration} seconds")
                    except Exception as e:
                        print(f"Error processing audio file {file_path}: {str(e)}")
                        return chunks
                else:
                    return chunks
        elif file_extension in audio_extensions:
            try:
                clip = AudioFileClip(str(file_path))  # Convert file_path to string
                duration = clip.duration
                print(f"Checked the audio file. Duration: {duration} seconds")
            except Exception as e:
                print(f"Error processing audio file {file_path}: {str(e)}")
                return chunks
        else:
            print(f"Unsupported file format: {file_path}")
            return chunks

        # If the file is larger than the max allowed size or duration, split into chunks
        if duration > max_duration:
            print("File duration exceeds limit. Checking file type and duration...")

            num_chunks = int(np.ceil(duration / max_duration))  # min(2,int(np.ceil(duration / max_duration))) # todo
            print(f"Duration exceeds limit. Splitting into {num_chunks} chunks...")

            for i in range(num_chunks):
                start_time = i * max_duration
                end_time = min((i + 1) * max_duration, duration)
                chunk_path = os.path.join(temp_folder, f"chunk_64_{i}.mp3")
                try:
                    clip.subclip(start_time, end_time).write_audiofile(chunk_path, bitrate="64k")
                except Exception as e:
                    print(f"Error transcribing file: {str(e)}")
                    m = traceback.format_exc()
                    print(m)
                    clip.subclip(start_time, end_time).audio.write_audiofile(chunk_path, bitrate="64k")

                chunks.append(chunk_path)
                print(f"Created chunk {i} from {start_time} to {end_time} seconds")

            clip.close()
        else:
            print("File duration is within limits")

            # Define start_time and end_time for the full clip
            start_time = 0
            end_time = duration

            # Define chunk_path for the full clip
            chunk_path = os.path.join(temp_folder, f"chunk_64_0.mp3")

            print(f"Converting the file {file_path} to 64k bitrate...")
            try:
                clip.subclip(start_time, end_time).write_audiofile(chunk_path, bitrate="64k")
            except Exception as e:
                print(f"Error transcribing file: {str(e)}")
                m = traceback.format_exc()
                print(m)
                clip.subclip(start_time, end_time).audio.write_audiofile(chunk_path, bitrate="64k")

            print(f"File converted successfully. New file path is {chunk_path}, {duration} seconds")
            chunks.append(chunk_path)

            clip.close()

        return chunks

    @staticmethod
    def format_time(elapsed_time: float) -> str:
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        time_str = ""
        if hours:
            time_str += f"{hours} hours "
        if minutes:
            time_str += f"{minutes} minutes "
        seconds = round(seconds)
        time_str += f"{seconds} seconds"

        return time_str.strip()




def timeformat_srt(time):
    hours = time // 3600
    minutes = (time - hours * 3600) // 60
    seconds = time - hours * 3600 - minutes * 60
    milliseconds = (time - int(time)) * 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"


def timeformat_vtt(time):
    hours = time // 3600
    minutes = (time - hours * 3600) // 60
    seconds = time - hours * 3600 - minutes * 60
    milliseconds = (time - int(time)) * 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}"


def write_file(subtitle, output_file):
    # Get the folder name from the output_file
    folder = os.path.dirname(output_file)
    # Check if the folder exists
    if not os.path.exists(folder):
        # Create the folder if not
        os.makedirs(folder)
    # Write the subtitle to the output_file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(subtitle)


def get_srt(segments):
    output = ""
    for i, segment in enumerate(segments):
        output += f"{i + 1}\n"
        output += f"{timeformat_srt(segment['start'])} --> {timeformat_srt(segment['end'])}\n"
        text = segment['text']
        if text.startswith(' '):
            text = text[1:]
        
        # 화자 정보가 있으면 추가
        if 'speaker' in segment and segment['speaker']:
            text = f"[{segment['speaker']}] {text}"
            
        output += f"{text}\n\n"
    return output


def get_vtt(segments):
    output = "WebVTT\n\n"
    for i, segment in enumerate(segments):
        output += f"{i + 1}\n"
        output += f"{timeformat_vtt(segment['start'])} --> {timeformat_vtt(segment['end'])}\n"
        text = segment['text']
        if text.startswith(' '):
            text = text[1:]
            
        # 화자 정보가 있으면 추가
        if 'speaker' in segment and segment['speaker']:
            text = f"[{segment['speaker']}] {text}"
            
        output += f"{text}\n\n"
    return output


def get_txt(segments):
    output = ""
    for i, segment in enumerate(segments):
        text = segment['text']
        if text.startswith(' '):
            text = text[1:]
            
        # 화자 정보가 있으면 추가
        if 'speaker' in segment and segment['speaker']:
            text = f"[{segment['speaker']}] {text}"
            
        output += f"{text}\n"
    return output


def convert_to_seconds(timestamp_str):
    # Split the string into start and end timestamps
    start, end = timestamp_str.split(' --> ')

    # Split the end timestamp into hours, minutes, and seconds + milliseconds
    end_h, end_m, end_s_ms = end.split(':')

    # Further split seconds and milliseconds
    end_s, end_ms = end_s_ms.split(',')

    # Convert to seconds
    end_time = float(end_h) * 3600 + float(end_m) * 60 + float(end_s) + float(end_ms) / 1000

    return end_time


def parse_srt_content(srt_data):
    """Reads SRT file and returns as dict"""
    # NoneType 오류 방지
    if not isinstance(srt_data, str) or not srt_data.strip():
        print("주의: 자막 내용이 비어있거나 유효하지 않습니다")
        return []
    
    try:    
        # 적절한 SRT 형식 확인
        srt_match = re.search(r"\d+\n\d{2}:\d{2}:\d{2},\d{3}", srt_data)
        if not srt_match:
            print("주의: SRT 포맷을 찾을 수 없습니다")
            return []
        
        # 유효한 부분만 추출
        srt_data = srt_data[srt_data.index(srt_match.group()):]
        
        data = []
        blocks = srt_data.split('\n\n')
        
        for block in blocks:
            if block.strip() != '':
                lines = block.strip().split('\n')
                if len(lines) >= 3:  # 유효한 자막 블록인지 확인
                    index = lines[0]
                    timestamp = lines[1]
                    sentence = ' '.join(lines[2:])
                    
                    data.append({
                        "index": index,
                        "timestamp": timestamp,
                        "sentence": sentence
                    })
        return data
    except Exception as e:
        print(f"자막 파싱 중 오류: {e}")
        return []


def parse_srt(file_path):
    """Reads SRT file and returns as dict"""
    with open(file_path, 'r', encoding='utf-8') as file:
        srt_data = file.read()

    data = []
    blocks = srt_data.split('\n\n')

    for block in blocks:
        if block.strip() != '':
            lines = block.strip().split('\n')
            index = lines[0]
            timestamp = lines[1]
            sentence = ' '.join(lines[2:])

            data.append({
                "index": index,
                "timestamp": timestamp,
                "sentence": sentence
            })
    return data


def parse_vtt(file_path):
    """Reads WebVTT file and returns as dict"""
    with open(file_path, 'r', encoding='utf-8') as file:
        webvtt_data = file.read()

    data = []
    blocks = webvtt_data.split('\n\n')

    for block in blocks:
        if block.strip() != '' and not block.strip().startswith("WebVTT"):
            lines = block.strip().split('\n')
            index = lines[0]
            timestamp = lines[1]
            sentence = ' '.join(lines[2:])

            data.append({
                "index": index,
                "timestamp": timestamp,
                "sentence": sentence
            })

    return data


def get_serialized_srt(dicts):
    output = ""
    for dic in dicts:
        output += f'{dic["index"]}\n'
        output += f'{dic["timestamp"]}\n'
        output += f'{dic["sentence"]}\n\n'
    return output


def get_serialized_vtt(dicts):
    output = "WebVTT\n\n"
    for dic in dicts:
        output += f'{dic["index"]}\n'
        output += f'{dic["timestamp"]}\n'
        output += f'{dic["sentence"]}\n\n'
    return output

def retype(title):
    return unicodedata.normalize('NFC', title).encode('utf-8').decode('utf-8')


def safe_filename(name):
    # safe_name = name

    #######
    INVALID_FILENAME_CHARS = r'[<>:"/\\|?*\x00-\x1f]'
    safe_name = retype(re.sub(INVALID_FILENAME_CHARS, '_', name))
    # Truncate the filename if it exceeds the max_length (20)
    safe_len = 35
    if len(safe_name) > safe_len:
        file_extension = safe_name.split('.')[-1]
        if len(file_extension) + 1 < safe_len:
            truncated_name = safe_name[:safe_len - len(file_extension) - 1]
            safe_name = truncated_name + '.' + file_extension
        else:
            safe_name = safe_name[:safe_len]

    return safe_name




def subtitle_translation(srt_dir="/content/drive/MyDrive/outputs"):

    try:
        # with open(f"/content/drive/MyDrive/auth.txt", "r") as f:
        #     auth_key = f.readline().strip()

        with open("/content/drive/MyDrive/auth.yaml", 'r') as file:
            config = yaml.safe_load(file)
            auth_key = config['auth_key'].strip()

    except:
        print("api key not found error.")
        auth_key = "x"

    # The target language code
    target_lang = "KO"

    def translate(text, target_lang):
        # The API endpoint
        endpoint = "https://api-free.deepl.com/v2/translate"

        # URL-encode the text
        text_encoded = urllib.parse.quote(text)

        # Construct the request URL
        url = endpoint + "?auth_key=" + auth_key + "&text=" + text_encoded + "&target_lang=" + target_lang

        # Send the request and get the response
        response = requests.get(url)

        # Parse the response as JSON
        data = response.json()

        # Extract the translated text
        translation = data["translations"][0]["text"]
        time.sleep(1)
        return translation

    def get_serialized_srt(dicts):
        output = ""
        for dic in dicts:
            output += f'{dic["index"]}\n'
            output += f'{dic["timestamp"]}\n'
            output += f'{dic["sentence"]}\n\n'
        return output

    def write_file(subtitle, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(subtitle)

    def display_truncated(text):
        if len(text) > 50:
            return text[:25] + "..." + text[-25:]
        else:
            return text

    def translate_text(t, mode='app'):
        return translate(t, "KO")

    def countdown(seconds):
        print(f'{seconds:.1f}초 휴식.')
        integer_seconds = int(seconds)
        decimal_seconds = seconds - integer_seconds
        for i in range(integer_seconds, 0, -1):
            print(f'{i}초 대기중..', end='\r')
            time.sleep(1)
        print(f'{decimal_seconds:.1f}초 대기중..', end='\r')
        time.sleep(decimal_seconds)


    def process_text_in_batches(d, batch_size=50):
        processed_list = []
        prev_msg = ''

        for i in range(0, len(d), batch_size):
            text_batch = '\n'.join([dic["sentence"] for dic in d[i:i + batch_size]])
            translated_batch = translate_text(text_batch).split('\n')

            msg = display_truncated('\n'.join(translated_batch))
            msg = msg + f'before len_translated_batch : {len(translated_batch)}\n'
            translated_batch = [text for text in translated_batch if 'Translated with deepL' not in text]
            msg = msg + f'after len_translated_batch : {len(translated_batch)}\n'

            is_expired = [text for text in translated_batch if
                          'DeepL이 마음에 드셨나요' in text or 'DeepL Pro를 사용해' in text or 'Pro 30일 무료 체험' in text]

            if len(is_expired) > 1 or prev_msg == msg:
                print('[deepL 사용한도 도달] keyword ' + msg)
                # input('시크릿 창을 다시 열고 엔터를 쳐주세요.') # 시크릿창 여는것도 자동화

                re_open_secret_window(driver)

                # 재작업
                translated_batch = translate_text(text_batch).split('\n')
                translated_batch = [text for text in translated_batch if 'Translated with deepL' not in text]
            else:
                print(msg)

            for j, dic in enumerate(d[i:i + batch_size]):
                dic["sentence"] = translated_batch[j]
            processed_list.extend(d[i:i + batch_size])

            prev_msg = msg

        return processed_list

    # srt_dir에서 .srt 파일 목록을 가져옵니다.
    srt_files = [f for f in os.listdir(srt_dir) if f.endswith('.srt') and '(ko)' not in f]

    # 각 .srt 파일에 대해 작업을 수행합니다.
    for si, srt_file in enumerate(srt_files):

        fname = os.path.join(srt_dir, srt_file)
        print(f"[{si}/{len(list(srt_files))}]")

        # 파일 경로에서 디렉토리와 파일명을 분리합니다.
        output_dir, file_with_ext = os.path.split(fname)
        # 파일명에서 확장자를 제거합니다.
        file_name, _ = os.path.splitext(file_with_ext)
        print("output_dir: ", output_dir)
        print("file_name: ", file_name)

        d = parse_srt_content(fname)

        processed_list = process_text_in_batches(d, 45)

        # 저장
        subtitle = get_serialized_srt(processed_list)
        output_path = f"{output_dir}/(ko){file_name}"

        write_file(subtitle, f"{output_path}.srt")

        # 원본 파일 삭제
        print(f'{fname} : translation completed, deleting... ')
        os.remove(fname)

    print(f'[자막 번역 완료]')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Subtitle Generation CLI')
    parser.add_argument('--mode', choices=['files', 'youtubes'], required=True,
                        help='Mode of operation: files or youtubes')
    parser.add_argument('--input_file_dir', type=str, help='Path to input directory for file mode')
    parser.add_argument('--youtube_links_input', type=str, help='Comma-separated YouTube links for youtubes mode')
    parser.add_argument('--model', type=str, default='large-v2', help='Whisper model size')
    parser.add_argument('--lang', type=str, default='english', help='Language for transcription and translation')
    parser.add_argument('--subformat', type=str, default='SRT', choices=['SRT', 'WebVTT', 'txt'],
                        help='Subtitle format')
    parser.add_argument('--translate', action='store_true', help='Enable translation to the target language')
    parser.add_argument('--force_skip_translation', action='store_true', 
                        help='Force translation even if the detected language is Korean')
    parser.add_argument('--use_diarization', action='store_true',
                        help='Enable speaker diarization (화자 분리)')
    return parser.parse_args()


def process_files(input_file_dir, model, lang, subformat, translate, use_diarization=False):
    whisper_inference = WhisperInference()
    # 화자 분리 기능 설정
    whisper_inference.use_diarization = use_diarization
    
    # Find all audio/video files in the given input directory
    file_extensions = ['*.mp3', '*.webm', '*.wav', '*.m4a', '*.mp4', '*.mkv']  # mkv 추가
    fl = list(itertools.chain(*(Path(input_file_dir).glob(ext) for ext in file_extensions)))

    # Load the YAML file and store it in a dictionary
    with open('config.yaml') as file:  # Replace 'config.yaml' with the name of your YAML file
        config = yaml.safe_load(file)  # Use safe_load to parse the YAML file
    my_drive_outputs = config['my_drive_outputs']

    for fi, audio_file in enumerate(fl):
        try:
            print(f'[{fi + 1}/{len(fl)}] Processing file: {audio_file}')

            # Temporary directory where chunks will be saved
            temp_folder = config['temp_folder']  # Use the value from the YAML file
            os.makedirs(temp_folder, exist_ok=True)

            # Split the file into chunks if necessary and convert to 64k bitrate
            audio_chunks = whisper_inference.check_and_split_audio(file_path=str(audio_file), temp_folder=temp_folder)

            final_subtitles = []
            last_timestamp = 0.0
            last_index = 0

            # Process chunks
            for chunk_idx, chunk_file in enumerate(audio_chunks):
                subtitle_content = whisper_inference.transcribe_file(
                    fileobjs=[chunk_file],
                    model_size=model,
                    beam_size=1,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    compute_type='float16',
                    lang=lang, istranslate=False,
                    # progress=None,
                    file_format=subformat,
                    add_timestamp=False,
                )

                # Add to the existing file
                parsed_subtitles = parse_srt_content(subtitle_content)
                if chunk_idx > 0:
                    adjusted_subtitles = whisper_inference.adjust_srt_timestamps(parsed_subtitles, last_timestamp,
                                                                                 last_index)
                else:
                    adjusted_subtitles = parsed_subtitles

                final_subtitles.extend(adjusted_subtitles)

                print(f'adjusted_subtitles = \n{adjusted_subtitles}')
                last_timestamp = convert_to_seconds(
                    adjusted_subtitles[-1]['timestamp']) if adjusted_subtitles else last_timestamp
                last_index = adjusted_subtitles[-1]['index'] if adjusted_subtitles else last_index

                # # Delete the processed chunk file
                # os.remove(chunk_file)

            # Save the final SRT file
            final_subtitles_str = whisper_inference.generate_srt_string(final_subtitles)
            file_name = audio_file.stem
            if lang == 'Korean' or lang == 'ko':
                file_name = '(ko)' + file_name
            output_srt_file_path = f"{file_name}.srt"
            try:
                print(f"Generating final SRT string...{my_drive_outputs}/{output_srt_file_path}")
                write_file(final_subtitles_str, f"{my_drive_outputs}/{output_srt_file_path}")

            except Exception as e:
                print(f"Error transcribing file: {str(e)}")
                m = traceback.format_exc()
                print(m)
                print('구글드라이브 백업 실패. 해당 경로를 확인하세요. ')

            # print(f"Opening file for writing... : {output_srt_file_path}")
            # with open(output_srt_file_path, 'w') as file:
            #     print("Writing to file...")
            #     file.write(final_subtitles_str)

            print("Process completed. Final SRT file saved.")

            # Delete the original file after chunks are processed
            os.remove(audio_file)
            print(f"{audio_file} : completed and deleted.")

        except Exception as e:
            print(f"Error transcribing file: {str(e)}")
            m = traceback.format_exc()
            print(m)
            print("process next file..")

def process_youtube_links(youtube_links_input, model, lang, subformat, translate, use_diarization=False):
    whisper_inference = WhisperInference()
    # 화자 분리 기능 설정
    whisper_inference.use_diarization = use_diarization
    
    youtube_links = youtube_links_input.split(',')
    for link in youtube_links:
        link = link.strip()
        try:
            whisper_inference.transcribe_youtube(
                youtubelink=link,
                model_size=model,
                lang=lang,
                beam_size=1,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                compute_type='float16',
                istranslate=translate,
                file_format=subformat,
                add_timestamp=True
            )
        except Exception as e:
            print(f"Error processing YouTube link {link}: {str(e)}")

def main():
    args = parse_arguments()
    if args.mode == 'files':
        if not os.path.isdir(args.input_file_dir):
            print(f'Error: The specified directory does not exist: {args.input_file_dir}')
            sys.exit(1)
        process_files(args.input_file_dir, args.model, args.lang, args.subformat, args.translate, args.use_diarization)
        if not args.force_skip_translation and args.lang != "Korean" and args.lang != "ko":
            print("Translation start")
            subtitle_translation()
    elif args.mode == 'youtubes':
        if not args.youtube_links_input:
            print('Error: No YouTube links provided for processing.')
            sys.exit(1)
        process_youtube_links(args.youtube_links_input, args.model, args.lang, args.subformat, args.translate, args.use_diarization)


if __name__ == '__main__':
    main()
