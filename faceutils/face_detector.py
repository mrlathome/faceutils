import dlib
import os
from .models import Face, Landmarks
from .distance import euclidean_distance

_base_dir = os.path.dirname(__file__)

_detector = dlib.get_frontal_face_detector()
_predictor = dlib.shape_predictor(
    os.path.join(os.path.join(_base_dir, "data"),
                 "shape_predictor_68_face_landmarks.dat")
)
_recognizer = dlib.face_recognition_model_v1(
    os.path.join(os.path.join(_base_dir, "data"),
                 "dlib_face_recognition_resnet_model_v1.dat")
)


def detect_faces(image, min_score=2, max_idx=2):
    output = []
    dets, scores, idx = _detector.run(image, 1, -1)
    for i, d in enumerate(dets):
        if scores[i] >= min_score and max_idx >= idx[i]:
            output.append(Face(d, scores[i]))
    return output


def face_landmarks(image, face):
    if not isinstance(face, Face):
        raise TypeError("face rect must be a instance of Face class")
    landmarks = _predictor(image, face.rect)
    return Landmarks(landmarks)


def extract_face_features(image, landmarks):
    return [float(i) for i in _recognizer.compute_face_descriptor(image, landmarks)]


def features_distance(features_a, features_b):
    return euclidean_distance(
        features_a,
        features_b,
        68,
    )
