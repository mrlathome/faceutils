import dlib


class Face:
    def __init__(self):
        self.rect = dlib.rectangle()
        self.score = 0.0
    
    def width(self):
        return self.rect.width()
    
    def height(self):
        return self.rect.height()

    def points(self):
        return (self.rect.left, self.rect.top), (self.rect.right, self.rect.bottom)

    def __str__(self):
        return "Rect: {}, Score: {}".format(self.rect, self.score)

    def __repr__(self):
        return self.__str__()
