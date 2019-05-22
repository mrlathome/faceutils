import cv2


def visualize(image, face, landmarks=None, color=(200, 100, 200), thickness=2):
    image_copy = image.copy()
    cv2.rectangle(image_copy, face.points()[0], face.points()[1], color, thickness)
    if landmarks is None:
        return image_copy

    def draw_lines(image, indices):
        for i in range(len(indices) - 1):
            first = landmarks.points[indices[i]]
            second = landmarks.points[indices[i+1]]
            cv2.line(image, first, second, color, thickness)
    
    draw_lines(image_copy, range(0, 17))
    draw_lines(image_copy, range(17, 22))
    draw_lines(image_copy, range(22, 27))
    draw_lines(image_copy, range(27, 31))
    draw_lines(image_copy, range(31, 36))
    draw_lines(image_copy, range(36, 42) + [36])
    draw_lines(image_copy, range(42, 48) + [42])
    draw_lines(image_copy, range(48, 60) + [48])
    return image_copy
