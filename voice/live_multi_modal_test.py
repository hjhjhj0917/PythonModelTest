import io
import json
import time
import requests
import librosa
import numpy as np
import torch
import pyaudio
import os

CLOVA_SPEECH_URL = "https://clovaspeech-gw.ncloud.com/external/v1/14749/57160681733f702f10e59522a59b57afd1c52db4c098d7d76d96b26c94797946"
CLOVA_SPEECH_KEY = "2808201d6b07456485ad0a01887c9f63"
CLOVA_VOICE_URL = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
NCP_CLIENT_ID = "wajzs8ql6c"
NCP_CLIENT_SECRET = "nkoKqPLA6llQ7Ym3EugAMrHhk5kLJprRcqOPYq26"

JSON_LOG_FILE = "health_analysis_log.json"

# VAD 모델 로드
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, _, read_audio, _, _) = utils

# 오디오 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512


# ==========================================
# 🛠️ 분석 결과 저장 및 재생 함수
# ==========================================

def save_analysis_json(data):
    existing_data = []
    if os.path.exists(JSON_LOG_FILE):
        with open(JSON_LOG_FILE, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except:
                existing_data = []

    existing_data.append(data)
    with open(JSON_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
    print(f"📄 분석 데이터(이벤트 포함) 저장 완료: {JSON_LOG_FILE}")


def speak_realtime(text):
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NCP_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NCP_CLIENT_SECRET,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"speaker": "nara", "speed": "0", "text": text}
    response = requests.post(CLOVA_VOICE_URL, headers=headers, data=data)
    if response.status_code == 200:
        temp_name = f"temp_res_{int(time.time())}.mp3"
        with open(temp_name, "wb") as f:
            f.write(response.content)
        os.system(f"start /min /wait {temp_name}")
        time.sleep(1)
        if os.path.exists(temp_name): os.remove(temp_name)

def run_realtime_system(cam_distance):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("\n🎤 [청취 중] 어르신의 말씀이나 상황을 감지하고 있습니다...")

    frames = []
    silent_chunks = 0
    is_speaking = False

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        speech_prob = model(torch.from_numpy(audio_chunk), RATE).item()

        if speech_prob > 0.4:  # 약간 더 민감하게 설정
            is_speaking = True
            silent_chunks = 0
        elif is_speaking:
            silent_chunks += 1

        if is_speaking and silent_chunks > (RATE / CHUNK * 1.2):  # 1.2초 대기
            break

    audio_data = b"".join(frames)
    y = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    # 1. 소리 크기 및 거리 보정
    raw_db = float(np.mean(librosa.amplitude_to_db(librosa.feature.rms(y=y))) + 85)
    corrected_db = round(raw_db + (20 * np.log10(cam_distance / 30)), 1)

    # 2. Clova Speech (이벤트 탐지 파라미터 강화)
    headers = {'X-CLOVASPEECH-API-KEY': CLOVA_SPEECH_KEY}
    params = {
        'language': 'ko-KR',
        'completion': 'sync',
        'noiseBreaker': True,
        # [핵심] 이벤트 탐지 활성화 및 모든 유형 지정
        'eventDetection': {
            'enable': True,
            'types': ['scream', 'cough', 'crying', 'clap']
        }
    }

    files = {
        'media': ('speech.wav', audio_data, 'audio/wav'),
        'params': (None, json.dumps(params), 'application/json')
    }

    try:
        response = requests.post(CLOVA_SPEECH_URL + '/recognizer/upload', headers=headers, files=files)
        stt_res = response.json()
        user_text = stt_res.get('text', '')
        # 탐지된 이벤트 추출
        detected_events = stt_res.get('events', [])
    except:
        user_text, detected_events = "", []

    # 3. 결과 분석 및 로그 생성
    event_list = [ev.get('label') for ev in detected_events]
    print(f"💬 인식: {user_text} | 🔔 탐지된 이벤트: {event_list}")

    analysis_result = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "metrics": {"corrected_db": corrected_db, "distance_cm": cam_distance},
        "extracted_text": user_text,
        "detected_events": detected_events,  # 이벤트 상세 정보(시간, 종류) 저장
        "is_emergency": "scream" in event_list  # 비명 감지 시 응급 플래그
    }
    save_analysis_json(analysis_result)

    # 4. 상황별 AI 응답
    if "scream" in event_list:
        ai_msg = "어르신! 비명 소리가 들렸어요. 어디 다치신 건 아니죠? 제가 도움을 요청할까요?"
    elif "cough" in event_list:
        ai_msg = "어르신, 기침 소리가 들리네요. 따뜻한 물 한 잔 드시는 게 좋겠어요."
    elif user_text:
        ai_msg = f"네 어르신, {user_text}라고 말씀하셨군요. 다 기록해 두었으니 걱정 마세요."
    else:
        ai_msg = "어르신, 몸 상태를 실시간으로 체크 중입니다. 편안히 계세요."

    speak_realtime(ai_msg)
    stream.stop_stream();
    stream.close();
    p.terminate()


if __name__ == "__main__":
    while True:
        try:
            run_realtime_system(cam_distance=70)
            time.sleep(1)
        except KeyboardInterrupt:
            break