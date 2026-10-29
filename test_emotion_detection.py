import unittest
from unittest.mock import Mock, patch

from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    def _mock_response(self, scores, status_code=200):
        response = Mock()
        response.status_code = status_code
        response.json.return_value = {
            "emotionPredictions": [{"emotion": scores}]
        }
        return response

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_dominant_joy(self, post):
        post.return_value = self._mock_response(
            {"anger": 0.01, "disgust": 0.01, "fear": 0.01,
             "joy": 0.95, "sadness": 0.02}
        )
        result = emotion_detector("I am glad this happened")
        self.assertEqual(result["dominant_emotion"], "joy")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_dominant_anger(self, post):
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
        post.return_value = self._mock_response(
            {"anger": 0.01, "disgust": 0.95, "fear": 0.01,
             "joy": 0.01, "sadness": 0.02}
        )
        self.assertEqual(emotion_detector("That is disgusting")["dominant_emotion"], "disgust")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_dominant_sadness(self, post):
        post.return_value = self._mock_response(
            {"anger": 0.01, "disgust": 0.01, "fear": 0.01,
             "joy": 0.02, "sadness": 0.95}
        )
        self.assertEqual(emotion_detector("I am sad")["dominant_emotion"], "sadness")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_dominant_fear(self, post):
        post.return_value = self._mock_response(
            {"anger": 0.01, "disgust": 0.01, "fear": 0.95,
             "joy": 0.01, "sadness": 0.02}
        )
        self.assertEqual(emotion_detector("I am afraid")["dominant_emotion"], "fear")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_status_400(self, post):
        post.return_value = self._mock_response({}, status_code=400)
        result = emotion_detector("bad request")
        self.assertIsNone(result["dominant_emotion"])
        self.assertIsNone(result["anger"])

    def test_blank_input(self):
        result = emotion_detector("   ")
        self.assertIsNone(result["dominant_emotion"])


if __name__ == "__main__":
    unittest.main()
