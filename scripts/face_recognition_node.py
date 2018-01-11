#!/usr/bin/env python
import cv2
import glob
import copy
import os
import pickle
import time
import sys
import json

import face_api
import config
import knn

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from face_recognition.msg import Box
from face_recognition.srv import Face, Name, NameResponse, FaceResponse

_topic = config.topic_name
_base_dir = os.path.dirname(__file__)
_face_dir = os.path.join(_base_dir, "faces")

# face_map is a place for objects
# with face id as key and face detail
# as value.
face_map = dict()

classifier = knn.Classifier(k=config.neighbors, thresh=config.dlib_face_threshold)


def name_controller(req):
    response = "FAILED"

    for key, value in face_map:
        if key == req.label:
            face_map[key]["name"] = req.name
            response = "OK"
            break

    return NameResponse(response)


class ImageReader:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(config.image_topic, Image, self.process)

        self.faces = []

        self.frame_limit = 0
        self.face_positions = []
        self.detected_faces = []
        self.known_faces = []

    def process(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            image = cv2.resize(cv_image, (0, 0), fx=config.scale, fy=config.scale)

            original = image.copy()

            if self.frame_limit % config.skip_frames == 0:
                self.frame_limit = 0
                # Detecting face positions
                self.face_positions = face_api.detect_faces(image, min_score=config.detection_score,
                                                            max_idx=config.detection_idx)
            self.frame_limit += 1

            if len(self.face_positions) > len(self.detected_faces):
                # Make the list empty
                self.detected_faces = []

                # Compute the 128D vector that describes the face in img identified by
                # shape.
                encodings = face_api.face_descriptor(image, self.face_positions)

                for face_position, encoding in zip(self.face_positions, encodings):
                    # Create object from face_position.
                    face = face_api.Face(face_position[0], tracker_timeout=config.tracker_timeout)

                    predicted_id = classifier.predict(encoding)
                    if predicted_id != 0:
                        face.details = face_map[predicted_id]
                    else:
                        face.details["gender"] = face_api.predict_gender(encoding)
                        face_map[face.details["id"]] = face.details

                    if face_map[face.details["id"]]["size"] < config.classification_size:
                        face_map[face.details["id"]]["size"] += 1

                        classifier.add_pair(encoding, face.details["id"])

                        face_path = os.path.join(_face_dir, face.details["id"])
                        if not os.path.exists(face_path):
                            os.mkdir(face_path)
                        with open(os.path.join(face_path, "{}.dump".format(int(time.time()))), 'wb') as fp:
                            pickle.dump(encoding, fp)

                    # Start correlation tracker for face.
                    face.tracker.start_track(image, face_position[0])

                    # Face detection score, The score is bigger for more confident
                    # detections.
                    rospy.loginfo(
                        "Face {}->{:5} , [{}] [score : {:.2f}]".format(face.details["id"],
                                                                       face.details["name"],
                                                                       face.details["gender"],
                                                                       face_position[1]))

                    self.detected_faces.append(face)
            else:
                for index, face in enumerate(self.detected_faces):
                    # Update tracker , if quality is low ,
                    # face will be removed from list.
                    if not face.update_tracker(image):
                        self.detected_faces.pop(index)

                    face.details = face_map[str(face.details["id"])]

                    face.draw_face(original)

            if config.show_window:
                cv2.imshow("image", original)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    rospy.signal_shutdown("q key pressed")

        except CvBridgeError as e:
            rospy.logerr(e)

    def service_controller(self, r):
        boxes = []
        for face in self.detected_faces:
            box = Box()
            box.x, box.y, box.w, box.h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
            box.is_known = face.details["name"] != face.details["id"]
            box.is_female = face.details["gender"] == "female"
            box.is_male = face.details["gender"] == "male"
            box.label = face.details["id"]
            box.scale = config.scale
            box.name = face.details["name"]
            boxes.append(box)

        response = FaceResponse(boxes)
        return response


def main():
    rospy.init_node(_topic, anonymous=True)

    try:
        show_window_param = rospy.search_param("show_window")
        tracker_quality_param = rospy.search_param("tracker_quality")
        scale_param = rospy.search_param("scale")
        if show_window_param is not None:
            config.show_window = rospy.get_param(show_window_param, config.show_window)
        if tracker_quality_param is not None:
            config.face_tracker_quality = int(rospy.get_param(tracker_quality_param, config.face_tracker_quality))
        if scale_param is not None:
            config.scale = float(rospy.get_param(scale_param, config.scale))
    except TypeError as err:
        rospy.logerr(err)

    image_reader = ImageReader()

    rospy.loginfo("Reading face database ...")
    # Load face_encodings from files.
    for parent_path in glob.glob(os.path.join(_face_dir, "*")):
        face_id = str(parent_path[-5:])
        if not face_id.isdigit():
            continue
        rospy.loginfo("Loading faces from {} directory ...".format(face_id))
        glob_list = glob.glob(os.path.join(parent_path, "*.dump"))
        for file_path in glob_list:
            print file_path, face_id
            with open(file_path, 'rb') as fp:
                face_encoding = pickle.load(fp)

            classifier.add_pair(face_encoding, face_id)
        rospy.loginfo("Directory {} with {} faces loaded.".format(face_id, len(glob_list)))

    # Load face names.
    faces_json_path = os.path.join(_face_dir, "faces.json")
    if os.path.exists(faces_json_path):
        with open(faces_json_path) as f:
            global face_map
            face_map = json.loads(f.read())

    rospy.loginfo("Listening to images reader")
    rospy.Service('/{}/faces'.format(_topic), Face, image_reader.service_controller)

    rospy.loginfo("Listening to names controller")
    rospy.Service('/{}/names_controller'.format(_topic), Name, name_controller)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.logwarn("Shutting done ...")
    finally:
        rospy.loginfo("Saving faces.json ...")
        with open(faces_json_path, 'w') as f:
            f.write(json.dumps(face_map))


if __name__ == '__main__':
    main()
