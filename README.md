## MRL @Home face recognition
> Face recognition using `dlib` and `kNN` classification with ROS compatibility.

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

---

### Face detector in a Docker image

Take a look at [facedetector](https://github.com/ahmdrz/facedetector). It's a simple Flask and docker-ready application that works as a server to detect faces, genders and their landmarks.

### Caution

The `Master` branch is heavily under development. Please use `v1` branch instead of `Master`.

<img src="https://github.com/mrlathome/faceutils/blob/master/resources/albert-einstein-2.jpg?raw=true" width="100%"/>

### Wiki

For details, please read our notes on [Wiki Pages](https://github.com/mrlathome/faceutils/wiki).

### Tests

```bash
$ sudo pip install nose
$ python2.7 -m nose -v --nocapture
```

### ROS module

To install it as ROS module, first of all, install `faceutils` with `pip` and then clone this repository to your `catkin_ws`. Do not forget to run `catkin_make`.

### Contribution

It's simple. Fork and work on it !

Remember to write a tests in `tests` directory before sending PR.

---

<img src="https://github.com/mrlathome/faceutils/blob/master/resources/image.jpg?raw=true" width="100%"/>

---

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Made in MRL](https://img.shields.io/badge/Made%20in-Mechatronic%20Research%20Labratories-red.svg)](https://www.qiau.ac.ir/)
