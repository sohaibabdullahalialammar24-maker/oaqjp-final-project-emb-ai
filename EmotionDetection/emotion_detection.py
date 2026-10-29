"""Emotion detection client for the Skills Network Watson NLP service."""

import requests

WATSON_EMOTION_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)


def _empty_result():
    """Return the required empty result for invalid input/service errors."""
    return {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }


def emotion_detector(text_to_analyze):
    """Return five emotion scores and the dominant emotion for input text."""
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return _empty_result()

    payload = {"raw_document": {"text": text_to_analyze}}
    try:
        response = requests.post(WATSON_EMOTION_URL, json=payload, timeout=15)
    except requests.RequestException:
        return _empty_result()

    if response.status_code != 200:
        return _empty_result()

    try:
        emotions = response.json()["emotionPredictions"][0]["emotion"]
        result = {
            "anger": emotions["anger"],
            "disgust": emotions["disgust"],
            "fear": emotions["fear"],
            "joy": emotions["joy"],
            "sadness": emotions["sadness"],
        }
    except (KeyError, IndexError, TypeError, ValueError):
        return _empty_result()

    result["dominant_emotion"] = max(
        ("anger", "disgust", "fear", "joy", "sadness"),
        key=lambda emotion: result[emotion],
    )
    return result
