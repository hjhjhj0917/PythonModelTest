from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from collections import defaultdict

MODEL_NAME = "searle-j/kote_for_easygoing_people"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    top_k=None,                  # 모든 라벨 반환
    function_to_apply="sigmoid",
    device=-1
)

# 일기 예시
diary = """
오늘 하루 종일 너무 지치고 힘들었다.
그래도 친구랑 얘기해서 조금 나아졌다.
근데 미래가 걱정된다.
"""

# 문장 분리
sentences = [s.strip() for s in diary.split("\n") if s.strip()]

threshold = 0.35
emotion_sum = defaultdict(float)
emotion_count = defaultdict(int)

for sentence in sentences:
    outputs = pipe(sentence)

    # 보통 [[{label, score}, ...]] 또는 [{label, score}, ...] 형태라서 둘 다 처리
    if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], list):
        outputs = outputs[0]

    print(f"\n문장: {sentence}")
    print("원본 결과:", outputs)

    for out in outputs:
        if out["score"] >= threshold:
            emotion_sum[out["label"]] += out["score"]
            emotion_count[out["label"]] += 1

# 평균 계산
final_results = []
for label in emotion_sum:
    avg_score = emotion_sum[label] / emotion_count[label]
    final_results.append({
        "emotion": label,
        "score": round(avg_score, 4),
        "count": emotion_count[label]
    })

# 정렬
final_results = sorted(final_results, key=lambda x: x["score"], reverse=True)

print("\n 일기 전체 감정 결과:")
for r in final_results[:10]:
    print(r)