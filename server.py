"""Flask web server for the Emotion Detector application."""

from flask import Flask, render_template, request

from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__)


@app.route("/")
def render_index_page():
    """Render the application page."""
    return render_template("index.html")


@app.route("/emotionDetector")
def emotion_detector_route():
    """Analyze query text and return a human-readable result."""
    text_to_analyze = request.args.get("textToAnalyze", "")
    if not text_to_analyze.strip():
        return "Invalid input! Please try again!", 400

    result = emotion_detector(text_to_analyze)
    if result["dominant_emotion"] is None:
        return "Emotion service is unavailable. Please try again later.", 502

    return (
        "For the given statement, the system response is "
        f"'anger': {result['anger']}, "
        f"'disgust': {result['disgust']}, "
        f"'fear': {result['fear']}, "
        f"'joy': {result['joy']} and "
        f"'sadness': {result['sadness']}. "
        f"The dominant emotion is {result['dominant_emotion']}."
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
