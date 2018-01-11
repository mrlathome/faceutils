#!/usr/bin/env python
import cv2
import dlib

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from face_recognition.srv import Face, Name, NameResponse, FaceResponse

import config
import face_api

_service = "/{}/faces".format(config.topic_name)


class ImageReader:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(config.image_topic, Image, self.process)

    def process(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            rospy.wait_for_service(_service)
            try:
                faces = rospy.ServiceProxy(_service, Face)
                resp1 = faces()
                faces = resp1.faces
                for f in faces:
                    rect = dlib.rectangle(
                        int(f.x / f.scale),
                        int(f.y / f.scale),
                        int((f.x + f.w) / f.scale),
                        int((f.y + f.h) / f.scale),
                    )
                    face = face_api.Face(rect)
                    face.details["id"] = f.label
                    face.details["name"] = f.name
                    gender = "unknown"
                    if f.is_male:
                        gender = "male"
                    if f.is_female:
                        gender = "female"
                    face.details["gender"] = gender

                    face.draw_face(image)
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e

            cv2.imshow("image", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                rospy.signal_shutdown("q key pressed")
            elif key == ord('s'):
                cv2.imwrite("output.jpg", image)

        except CvBridgeError as e:
            rospy.logerr(e)


def main():
    rospy.init_node(config.topic_name, anonymous=True)

    rospy.loginfo("Listening to images reader")
    ImageReader()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.logwarn("Shutting done ...")


if __name__ == "__main__":
    main()
