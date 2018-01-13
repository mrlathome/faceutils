# Only classification_size will save in files and memory.
classification_size = 10

# Reduce resolution with this scale.
scale = 0.95

# Each face_tracker will die after timeout.
# Timeout is in seconds.
tracker_timeout = 5

# The score is bigger for more confident detections.
# TODO: Find best score.
# In my case , 2.0 is the best detection score.
# Maximum score that I saw is 2.7
detection_score = 1.0

# Maximum face detection score
max_face_score = 2.6

# This can be used to broadly identify faces in different orientations.
detection_idx = 3.0

# Each skip_frames times , face detection will start.
# With this option we can make it more fast.
# For low scale (for example 0.5 or less than 0.5)
# we don't need skip_frames , So it can be 1.
skip_frames = 2

# ROS topic name.
topic_name = "ros_face_recognition"

# ROS topic that contains video stream.
image_topic = "/usb_cam/image_raw"

# Tracker quality of each face.
face_tracker_quality = 8

# Show window
show_window = False

# neighbors in kNN classification
neighbors = 3

# dlib_face_threshold , as euclidean distance
dlib_face_threshold = 0.6
