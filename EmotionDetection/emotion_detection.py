"""Client for the Skills Network Watson NLP emotion service."""

import requests

WATSON_EMOTION_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
WATSON_HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}
EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")


def _empty_result():
    """Return an empty, consistently shaped result when input/service data is unavailable."""
    return {
        **{emotion: None for emotion in EMOTIONS},
        "dominant_emotion": None,
    }


def emotion_detector(text_to_analyze):
    """Return five Watson emotion scores and their dominant label for input text."""
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return _empty_result()

    payload = {"raw_document": {"text": text_to_analyze}}
    try:
        response = requests.post(
            WATSON_EMOTION_URL,
            json=payload,
            headers=WATSON_HEADERS,
            timeout=15,
        )
    except requests.RequestException:
        return _empty_result()

    if response.status_code != 200:
        return _empty_result()

    try:
        scores = response.json()["emotionPredictions"][0]["emotion"]
        result = {emotion: scores[emotion] for emotion in EMOTIONS}
    except (KeyError, IndexError, TypeError, ValueError):
        return _empty_result()

    if any(not isinstance(score, (int, float)) for score in result.values()):
        return _empty_result()

    result["dominant_emotion"] = max(EMOTIONS, key=result.get)
    return result
