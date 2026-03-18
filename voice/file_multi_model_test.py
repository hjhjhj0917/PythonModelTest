import io
import json
import requests
import librosa
import numpy as np
import os
import time

CLOVA_SPEECH_URL = "https://clovaspeech-gw.ncloud.com/external/v1/14749/57160681733f702f10e59522a59b57afd1c52db4c098d7d76d96b26c94797946"
CLOVA_SPEECH_KEY = "2808201d6b07456485ad0a01887c9f63"
CLOVA_VOICE_URL = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
NCP_CLIENT_ID = "wajzs8ql6c"
NCP_CLIENT_SECRET = "nkoKqPLA6llQ7Ym3EugAMrHhk5kLJprRcqOPYq26"

JSON_LOG_FILE = "health_analysis_log.json"

def save_analysis_json(data):
    """ 분석 결과를 JSON 파일에 누적 저장 (파일 기반 DB 대용) """
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
    print(f"📄 분석 결과가 {JSON_LOG_FILE}에 안전하게 저장되었습니다.")


def play_audio_safely(audio_bytes):
    """ 재생 후 즉시 파기 """
    temp_filename = f"temp_res_{int(time.time())}.mp3"
    with open(temp_filename, "wb") as f:
        f.write(audio_bytes)
    os.system(f"start /min /wait {temp_filename}")
    time.sleep(1)
    if os.path.exists(temp_filename):
        os.remove(temp_filename)


def run_advanced_analysis(file_path, cam_distance):
    print(f"🔍 [소음 분리 분석 시작] {file_path}")

    # 1. 음향 수치 추출
    y, sr = librosa.load(file_path, sr=16000)
    raw_db = float(np.mean(librosa.amplitude_to_db(librosa.feature.rms(y=y))) + 85)
    corrected_db = round(raw_db + (20 * np.log10(cam_distance / 30)), 1)

    # 2. Clova Speech (배경 소음 억제 및 화자 분리 설정)
    with open(file_path, 'rb') as f:
        audio_binary = f.read()

    headers = {'X-CLOVASPEECH-API-KEY': CLOVA_SPEECH_KEY}
    # [차별점] boost: 배경 소음 대비 목소리 강조 설정 적용
    params = {
        'language': 'ko-KR',
        'completion': 'sync',
        'noiseBreaker': True,  # 배경 소음 제거
        'diarization': {
            'enable': True,  # 화자 분리 활성화
            'speakerCountMin': 1,
            'speakerCountMax': 2
        },
        'fullText': True
    }

    files = {
        'media': (file_path, audio_binary, 'audio/mpeg'),
        'params': (None, json.dumps(params), 'application/json')
    }

    response = requests.post(CLOVA_SPEECH_URL + '/recognizer/upload', headers=headers, files=files)
    stt_res = response.json()
    print("DEBUG API RESPONSE:", json.dumps(stt_res, indent=2, ensure_ascii=False))
    user_text = stt_res.get('text', '')

    # 3. 분석 결과 객체 생성 (음성 파일 대신 저장할 데이터)
    analysis_result = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "input_file_source": file_path,
        "metrics": {
            "raw_db": round(raw_db, 1),
            "corrected_db": corrected_db,
            "distance_cm": cam_distance
        },
        "extracted_text": user_text,
        "stt_confidence": np.mean([s.get('confidence', 0) for s in stt_res.get('segments', [])]) if stt_res.get(
            'segments') else 0,
        "ai_opinion": "정상 범위 발화" if corrected_db > 60 else "발성 약화(주의)"
    }

    # JSON 저장 및 출력
    save_analysis_json(analysis_result)

    # 4. 응답 (TTS)
    tts_headers = {
        "X-NCP-APIGW-API-KEY-ID": NCP_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NCP_CLIENT_SECRET,
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {
        "speaker": "nara",
        "text": f"어르신, {user_text}라고 말씀하셨군요. 분석 결과를 저장했습니다."
    }

    res_voice = requests.post(CLOVA_VOICE_URL, headers=tts_headers, data=data)
    if res_voice.status_code == 200:
        play_audio_safely(res_voice.content)


if __name__ == "__main__":
    TEST_FILE = "test_speech.mp3"
    if os.path.exists(TEST_FILE):
        run_advanced_analysis(TEST_FILE, 70)