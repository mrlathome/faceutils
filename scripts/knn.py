import math
import operator


class Classifier:
    def __init__(self, k=3, thresh=0.6):
        self.k = k
        self.thresh = thresh
        self._samples = []
        self._labels = []

    def add_pair(self, sample, label):
        self._samples.append(sample)
        self._labels.append(label)

    @staticmethod
    def euclidean_distance(instance1, instance2, length):
        distance = 0
        for x in range(length):
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)

    def predict(self, target):
        if len(self._labels) == 0:
            return 0

        distances = []
        length = len(target)
        for x in range(len(self._samples)):
            dist = self.euclidean_distance(target, self._samples[x], length)
            if dist < self.thresh:
                distances.append((self._samples[x], dist, self._labels[x]))

        if not distances:
            return 0

        distances.sort(key=operator.itemgetter(1))
        n = self.k
        if n > len(distances):
            n = len(distances)

        class_votes = dict()
        for x in range(n):
            response = distances[x][2]
            if response in class_votes:
                class_votes[response] += 1
            else:
                class_votes[response] = 1
        sorted_otes = sorted(class_votes.iteritems(), key=operator.itemgetter(1), reverse=True)

        return sorted_otes[0][0]
