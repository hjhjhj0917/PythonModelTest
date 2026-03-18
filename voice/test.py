import requests
import json
import os

# --- 설정 정보 (NCP 콘솔에서 확인한 값 입력) ---
CLOVA_SPEECH_URL = "https://clovaspeech-gw.ncloud.com/external/v1/14749/57160681733f702f10e59522a59b57afd1c52db4c098d7d76d96b26c94797946"  # 호출 URL
CLOVA_SPEECH_KEY = "2808201d6b07456485ad0a01887c9f63"  # 시크릿 키


def run_stt_test(file_path):
    print(f"🚀 STT 테스트 시작: {file_path}")

    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return

    # 1. 파일 읽기
    with open(file_path, 'rb') as f:
        audio_binary = f.read()

    # 2. API 헤더 및 파라미터 설정
    headers = {
        'X-CLOVASPEECH-API-KEY': CLOVA_SPEECH_KEY
    }

    # 화자 분리(Diarization) 활성화 설정
    params = {
        'language': 'ko-KR',
        'completion': 'sync',  # 결과가 나올 때까지 대기
        'noiseBreaker': True,  # 배경 소음 제거
        'diarization': {
            'enable': True,
            'speakerCountMin': 1,
            'speakerCountMax': 5  # 테스트용으로 5명까지 허용
        },
        'fullText': True
    }

    files = {
        'media': (os.path.basename(file_path), audio_binary, 'audio/mpeg'),
        'params': (None, json.dumps(params), 'application/json')
    }

    # 3. API 호출
    print("📡 Clova Speech 서버로 요청을 보내는 중...")
    response = requests.post(
        f"{CLOVA_SPEECH_URL}/recognizer/upload",
        headers=headers,
        files=files
    )

    if response.status_code == 200:
        res_data = response.json()

        print("\n" + "=" * 50)
        print("📢 [STT 및 화자 분리 결과]")
        print("=" * 50)

        # 전체 텍스트 출력
        print(f"📝 전체 텍스트: {res_data.get('text')}\n")

        # 화자별 세부 대화 내용 출력
        segments = res_data.get('segments', [])
        print("💬 화자별 대화 상세:")
        for seg in segments:
            speaker = seg.get('speaker', {}).get('label', '알 수 없음')
            text = seg.get('text')
            start_time = seg.get('start') / 1000  # 초 단위 변환

            print(f"[{start_time:>5.1f}s] 화자 {speaker}: {text}")

        print("=" * 50)
        print(f"✅ 테스트 완료 (감지된 화자 수: {len(res_data.get('speakers', []))}명)")

    else:
        print(f"❌ API 호출 실패 (상태 코드: {response.status_code})")
        print(f"오류 내용: {response.text}")


# --- 실행부 ---
if __name__ == "__main__":
    # 테스트할 오디오 파일명을 입력하세요 (mp3, wav 등)
    TEST_AUDIO_FILE = "test_speech.mp3"

    run_stt_test(TEST_AUDIO_FILE)