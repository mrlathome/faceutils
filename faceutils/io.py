import cv2


def load_image(path):
    return cv2.imread(path)


def save_image(path, image):
    cv2.imwrite(path, image)
