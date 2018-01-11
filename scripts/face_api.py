import dlib
import os
import cv2
import random
import time
import pickle

_base_dir = os.path.dirname(__file__)
_data_dir = os.path.join(_base_dir, "data")

_detector = dlib.get_frontal_face_detector()
_predictor = dlib.shape_predictor(
    os.path.join(os.path.join(_base_dir, "data"), "shape_predictor_68_face_landmarks.dat"))
_recognizer = dlib.face_recognition_model_v1(
    os.path.join(os.path.join(_base_dir, "data"), "dlib_face_recognition_resnet_model_v1.dat"))

_classifier = pickle.load(open(os.path.join(os.path.join(_base_dir, "data"), "gender_model.pickle"), 'r'))


def predict_gender(encoding):
    result = _classifier(dlib.vector(encoding))
    if result > 0.5:
        return "male"

    if result < -0.5:
        return "female"


class Tracker:
    def __init__(self, timeout=None):
        self.tracker = dlib.correlation_tracker()
        self.rect = None
        self.timeout = timeout
        self.start_time = time.time()

    def start_track(self, image, rect):
        self.tracker.start_track(image, rect)

    def update_tracker(self, image, min_quality=8):
        if self.timeout:
            if time.time() - self.timeout > self.start_time:
                return False
        quality = self.tracker.update(image)
        if quality > min_quality:
            pos = self.tracker.get_position()
            self.rect = dlib.rectangle(
                int(pos.left()),
                int(pos.top()),
                int(pos.right()),
                int(pos.bottom()),
            )
            return True

        return False


class Face:
    def __init__(self, rect, tracker_timeout=10):
        self.rect = rect
        self.tracker = Tracker(timeout=tracker_timeout)
        tmp_id = str(random.randrange(10000, 99999))
        self.details = {"id": tmp_id, "gender": "unknown", "name": tmp_id, "size": 0}

    def update_tracker(self, image, min_quality=5):
        if self.tracker.update_tracker(image, min_quality):
            self.rect = self.tracker.rect
            return True
        return False

    def draw_face(self, image, scale=0.15):
        scale_x = (self.rect.right() - self.rect.left()) / 4
        scale_y = (self.rect.bottom() - self.rect.top()) / 4
        cv2.line(image, (self.rect.left(), self.rect.top()), (self.rect.left() + scale_x, self.rect.top()),
                 (100, 200, 100))
        cv2.line(image, (self.rect.left(), self.rect.top()), (self.rect.left(), self.rect.top() + scale_y),
                 (100, 200, 100))
        cv2.line(image, (self.rect.right(), self.rect.top()), (self.rect.right() - scale_x, self.rect.top()),
                 (100, 200, 100))
        cv2.line(image, (self.rect.right(), self.rect.top()), (self.rect.right(), self.rect.top() + scale_y),
                 (100, 200, 100))
        cv2.line(image, (self.rect.left(), self.rect.bottom()), (self.rect.left() + scale_x, self.rect.bottom()),
                 (100, 200, 100))
        cv2.line(image, (self.rect.left(), self.rect.bottom()), (self.rect.left(), self.rect.bottom() - scale_y),
                 (100, 200, 100))
        cv2.line(image, (self.rect.right(), self.rect.bottom()),
                 (self.rect.right() - scale_x, self.rect.bottom()), (100, 200, 100))
        cv2.line(image, (self.rect.right(), self.rect.bottom()),
                 (self.rect.right(), self.rect.bottom() - scale_y), (100, 200, 100))

        font_scale = scale_x / (image.shape[1] * scale)

        text = "{} {}".format(self.details["name"], self.details["gender"])

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        cv2.rectangle(image, (self.rect.left(), self.rect.top()),
                      (self.rect.right(), int(self.rect.top() - text_size[0][1] - 10)),
                      (100, 200, 100), -1)
        cv2.putText(image, text, (self.rect.left() + 5, self.rect.top() - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (50, 50, 255),
                    1, cv2.LINE_AA)


def detect_faces(img, min_score=2, max_idx=2):
    output = []
    # The third argument to run is an optional adjustment to the detection threshold,
    # where a negative value will return more detections and a positive value fewer.
    # Also, the idx tells you which of the face sub-detectors matched.  This can be
    # used to broadly identify faces in different orientations.
    dets, scores, idx = _detector.run(img, 1, -1)
    for i, d in enumerate(dets):
        if scores[i] >= min_score and max_idx >= idx[i]:
            output.append([d, scores[i]])
    return output


def face_descriptor(img, rectangles):
    return [_recognizer.compute_face_descriptor(img, _predictor(img, rect[0]), 1) for
            rect in rectangles]


def cluster_faces(descriptors, threshold=0.45):
    return dlib.chinese_whispers_clustering(descriptors, threshold)
