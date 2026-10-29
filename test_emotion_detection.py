"""Unit tests for the packaged emotion detector using mocked HTTP responses."""

import unittest
from unittest.mock import Mock, patch

from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    """Tests for dominant emotion formatting and invalid responses."""

    def _mock_response(self, scores, status_code=200):
        """Build a mock response with the Watson NLP response shape."""
        response = Mock()
        response.status_code = status_code
        response.json.return_value = {
            "emotionPredictions": [{"emotion": scores}]
        }
        return response

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_dominant_joy(self, post):
        """Select joy as dominant and return all five emotion scores."""
        post.return_value = self._mock_response(
            {"anger": 0.01, "disgust": 0.01, "fear": 0.01,
             "joy": 0.95, "sadness": 0.02}
        )
        result = emotion_detector("I am glad this happened")
        self.assertEqual(result["dominant_emotion"], "joy")
        self.assertEqual(
            set(result),
            {"anger", "disgust", "fear", "joy", "sadness", "dominant_emotion"},
        )

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_dominant_anger(self, post):
        """Select anger as the dominant emotion."""
        post.return_value = self._mock_response(
            {"anger": 0.95, "disgust": 0.01, "fear": 0.01,
             "joy": 0.01, "sadness": 0.02}
        )
        self.assertEqual(
            emotion_detector("I am really mad about this")["dominant_emotion"],
            "anger",
        )

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_dominant_disgust(self, post):
        """Select disgust as the dominant emotion."""
        post.return_value = self._mock_response(
            {"anger": 0.01, "disgust": 0.95, "fear": 0.01,
             "joy": 0.01, "sadness": 0.02}
        )
        self.assertEqual(
            emotion_detector("That is disgusting")["dominant_emotion"],
            "disgust",
        )

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_dominant_sadness(self, post):
        """Select sadness as the dominant emotion."""
        post.return_value = self._mock_response(
            {"anger": 0.01, "disgust": 0.01, "fear": 0.01,
             "joy": 0.02, "sadness": 0.95}
        )
        self.assertEqual(
            emotion_detector("I am sad")["dominant_emotion"],
            "sadness",
        )

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_dominant_fear(self, post):
        """Select fear as the dominant emotion."""
        post.return_value = self._mock_response(
            {"anger": 0.01, "disgust": 0.01, "fear": 0.95,
             "joy": 0.01, "sadness": 0.02}
        )
        self.assertEqual(
            emotion_detector("I am afraid")["dominant_emotion"],
            "fear",
        )

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_status_400(self, post):
        """Return the empty result shape for an HTTP 400 response."""
        post.return_value = self._mock_response({}, status_code=400)
        result = emotion_detector("bad request")
        self.assertIsNone(result["dominant_emotion"])
        self.assertIsNone(result["anger"])

    def test_blank_input(self):
        """Return the empty result without calling Watson for blank text."""
        result = emotion_detector("   ")
        self.assertIsNone(result["dominant_emotion"])


if __name__ == "__main__":
    unittest.main()
