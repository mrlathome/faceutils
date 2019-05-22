import os
import unittest
import tempfile
import shutil

from faceutils import detect_faces, io, face_landmarks, extract_face_features, features_distance


class RecognitionTest(unittest.TestCase):
    """
    Test methods in face recognition
    """

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def get_features(self, image):
        return extract_face_features(image, face_landmarks(image, detect_faces(image, min_score=1)[0]).shape)

    def test_features(self):
        image_a = io.load_image("faces/albert-einstein.jpg")
        image_b = io.load_image("faces/einstein-laughing.jpg")
        features_a = self.get_features(image_a)
        features_b = self.get_features(image_b)
        distance = features_distance(features_a, features_b)
        self.assertTrue(distance < 0.6)


if __name__ == '__main__':
    unittest.main()
