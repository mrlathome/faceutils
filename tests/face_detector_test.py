import os
import unittest
import tempfile
import shutil

from faceutils import detect_faces, io


class DetectorTest(unittest.TestCase):
    """
    Test methods in face detector
    """

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_einstein(self):
        image = io.load_image("faces/albert-einstein.jpg")
        faces = detect_faces(image)        
        self.assertTrue(len(faces) == 1)
        self.assertTrue(faces[0].width() == 156)


if __name__ == '__main__':
    unittest.main()