import os
import unittest
import tempfile
import shutil

from faceutils import detect_faces, io, face_landmarks


class LandamrksTest(unittest.TestCase):
    """
    Test methods in face landmarks detector
    """

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_landmarks(self):
        image = io.load_image("faces/albert-einstein.jpg")
        faces = detect_faces(image)
        self.assertTrue(len(faces) == 1)
        face = faces[0]
        landmarks = face_landmarks(image, face)
        self.assertTrue(len(landmarks) == 68)


if __name__ == '__main__':
    unittest.main()
