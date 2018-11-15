import dlib
from .models import Face

_detector = dlib.get_frontal_face_detector()


def detect_faces(image, min_score=2, max_idx=2):
    output = []
    dets, scores, idx = _detector.run(image, 1, -1)
    for i, d in enumerate(dets):
        if scores[i] >= min_score and max_idx >= idx[i]:
            face = Face()
            face.rect = d
            face.score = scores[i]
            output.append(face)
    return output
