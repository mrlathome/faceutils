import dlib


class Face:
    def __init__(self, rect, score=0.0):
        self.rect = rect
        self.score = score

    def width(self):
        return self.rect.width()

    def height(self):
        return self.rect.height()

    def points(self):
        return (int(self.rect.left()), int(self.rect.top())), (int(self.rect.right()), int(self.rect.bottom()))

    def __str__(self):
        return "Rect: {}, Score: {}".format(self.rect, self.score)

    def __repr__(self):
        return self.__str__()


class Landmarks:
    def __init__(self, shape):
        self.points = [(shape.part(i).x, shape.part(i).y)
                          for i in range(68)]
        self.shape = shape    

    def __len__(self):
        return len(self.points)
