
# MRL @Home face recognition

Face recognition using dlib and optimized kNN classification in ROS.


----------


![screenshot of MRL@HOME's face recognition](https://github.com/mrl-athomelab/ros-face-recognition/blob/master/resources/screenshot.png?raw=true)

# Contents
* Face detection
* Face tracker
* Gender detection
* ROS module
* Classification using kNN

## Face detection

In our program , We use [dlib](http://dlib.net) to detect faces in each frame. This is a simple part of our program.

from `face_api.py` module.
```python
import dlib
...
_detector = dlib.get_frontal_face_detector()
...
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
```

`_detector` is created using the `scan_fhog_pyramid` object from `dlib` , it's an instance of `object_detector`. This object is a tool for detecting the positions of objects (in our case , face) in an image. ( [read more](http://dlib.net/imaging.html#object_detector) )

According to dlib notes :
>This face detector is made using the now classic Histogram of Oriented Gradients (HOG) feature combined with a linear classifier, an image pyramid, and sliding window detection scheme.  This type of object detector is fairly general and capable of detecting many types of semi-rigid objects in addition to human faces.

With `dlib` face detector , We can find bounding rectangle of faces in each frames.

## Face tracker

Now we found faces in image , It's time to track faces.

#### Why we need to track faces ?

> If you have ever played with OpenCV face detection, you know that it works in real time and you can easily detect the face in every frame. So, why do you need tracking in the first place? Let’s explore the different reasons you may want to track objects in a video and not just do repeated detections.

**1. Tracking is faster than Detection**

> Usually tracking algorithms are faster than detection algorithms. The reason is simple. When you are tracking an object that was detected in the previous frame, you know a lot about the appearance of the object. You also know the location in the previous frame and the direction and speed of its motion. So in the next frame, you can use all this information to predict the location of the object in the next frame and do a small search around the expected location of the object to accurately locate the object. A good tracking algorithm will use all information it has about the object up to that point while a detection algorithm always starts from scratch. Therefore, while designing an efficient system usually an object detection is run on every nth frame while the tracking algorithm is employed in the n-1 frames in between. Why don’t we simply detect the object in the first frame and track subsequently? It is true that tracking benefits from the extra information it has, but you can also lose track of an object when they go behind an obstacle for an extended period of time or if they move so fast that the tracking algorithm cannot catch up. It is also common for tracking algorithms to accumulate errors and the bounding box tracking the object slowly drifts away from the object it is tracking. To fix these problems with tracking algorithms, a detection algorithm is run every so often. Detection algorithms are trained on a large number of examples of the object. They, therefore, have more knowledge about the general class of the object. On the other hand, tracking algorithms know more about the specific instance of the class they are tracking.

**2. Tracking can help when detection fails**

>  If you are running a face detector on a video and the person’s face get’s occluded by an object, the face detector will most likely fail. A good tracking algorithm, on the other hand, will handle some level of occlusion. In the video below, you can see Dr. Boris Babenko, the author of the MIL tracker, demonstrate how the MIL tracker works under occlusion.

**3. Tracking preserves identity**

> The output of object detection is an array of rectangles that contain the object. However, there is no identity attached to the object. For example, in the video below, a detector that detects red dots will output rectangles corresponding to all the dots it has detected in a frame. In the next frame, it will output another array of rectangles. In the first frame, a particular dot might be represented by the rectangle at location 10 in the array and in the second frame, it could be at location 17. While using detection on a frame we have no idea which rectangle corresponds to which object. On the other hand, tracking provides a way to literally connect the dots!

---

Here is a example of `BOOSTING tracker` in `OpenCV`  , face_follower :

```python
import cv2
import dlib

camera = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

old_faces = []

while True:
    ret, image = camera.read()
    if not ret:
        break

    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    faces = detector(image, 1)
    if len(old_faces) < len(faces):
        old_faces = []
        for face in faces:
            tracker = cv2.Tracker_create('BOOSTING')
            box = (face.left(), face.top(), face.width(), face.height())
            tracker.init(image, box)
            old_faces.append(tracker)
    else:
        for i, tracker in enumerate(old_faces):
            ok, bbox = tracker.update(image)
            if ok:
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                              (100, 200, 100))
            else:
                old_faces.pop(i)

    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()
```

In this example , If you run it , And move your face fastly , moving in depth or many other motions , Tracker will fail !

---

We use `correlation tracker` in our Robot ( We named it Robina ) but why we choose it instead of `OpenCV trackers` ? Simply !

Correlation tracker has these features :
1. Fast
2. Trustworthy
3. High accuracy ( winner of [VOT2014](http://www.votchallenge.net/vot2014/download/vot_2014_presentation.pdf) )

Here is an example of correlation tracker implemented in dlib :

```python
import cv2
import dlib

camera = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

old_faces = []

while True:
    ret, image = camera.read()
    if not ret:
        break

    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    faces = detector(image, 1)
    if len(old_faces) < len(faces):
        old_faces = []
        for face in faces:
            tracker = dlib.correlation_tracker()
            tracker.start_track(image, face)
            old_faces.append(tracker)
    else:
        for i, tracker in enumerate(old_faces):
            quality = tracker.update(image)
            if quality > 7:
                pos = tracker.get_position()
                pos = dlib.rectangle(
                    int(pos.left()),
                    int(pos.top()),
                    int(pos.right()),
                    int(pos.bottom()),
                )
                cv2.rectangle(image, (pos.left(), pos.top()), (pos.right(), pos.bottom()),
                              (100, 200, 100))
            else:
                old_faces.pop(i)

    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
```

Now , Time to see differences :

| Dlib correlation | OpenCV MIL tracker |
|--|--|
| ![dlib correlaion tracker](https://github.com/mrl-athomelab/ros-face-recognition/blob/master/resources/dlib_correlation.gif?raw=true) | ![opencv mil tracker](https://github.com/mrl-athomelab/ros-face-recognition/blob/master/resources/opencv_mil.gif?raw=true)  |


## Gender detection

...
