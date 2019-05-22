import os
import unittest
import tempfile
import shutil
import cv2
from glob import glob

from faceutils import detect_faces, io, visualize, face_landmarks


class VisualizeTest(unittest.TestCase):
    """
    Test methods in face visualizer
    """

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_faces(self):
        for image_file in glob("faces/*.jpg"):
            image = io.load_image(image_file)
            faces = detect_faces(image, min_score=1.0)
            face = faces[0]
            landmarks = face_landmarks(image, face)
            image = visualize(
                image,
                face,
                landmarks=landmarks,
                thickness=1,
                color=(0, 0, 255),
            )
            cv2.imshow('image', image)
            cv2.imwrite('images/'+image_file, image)
            cv2.waitKey(0)


if __name__ == '__main__':
    unittest.main()
