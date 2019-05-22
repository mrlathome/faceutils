import os
import unittest
import tempfile
import shutil

from faceutils import detect_faces, io, face_landmarks, extract_face_features, features_distance


class FeaturesTest(unittest.TestCase):
    """
    Test methods in face features extractor
    """

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_features(self):
        image = io.load_image("faces/albert-einstein.jpg")
        faces = detect_faces(image)
        self.assertTrue(len(faces) == 1)
        face = faces[0]
        landmarks = face_landmarks(image, face)
        features = extract_face_features(image, landmarks.shape)
        self.assertTrue(len(features) == 128)


if __name__ == '__main__':
    unittest.main()
