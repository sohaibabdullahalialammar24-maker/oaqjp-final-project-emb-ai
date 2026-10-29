# Final Project: Emotion Detection Application

A Flask web application that analyzes text with the Skills Network Watson NLP emotion service and returns scores for anger, disgust, fear, joy, and sadness, plus the dominant emotion.

## Run locally

```bash
python3 -m pip install -r requirements.txt
python3 server.py
```

Open `http://127.0.0.1:5000` and enter text in the form. The application sends requests to the live Skills Network Watson NLP endpoint; a reachable network and an available service are required for successful analysis. Empty input is rejected with HTTP 400. An unavailable upstream emotion service is reported with HTTP 502.

## Tests and static analysis

```bash
python3 -m unittest -v
python3 -m pylint EmotionDetection emotion_detection.py server.py test_emotion_detection.py
```

The unit tests mock the HTTP response to test parsing, dominant-emotion selection, and error handling. They do not prove that the live Watson endpoint is reachable or return real Watson scores. A live-service test must be run separately in a connected Skills Network environment.
