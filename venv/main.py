import pandas as pd
import math as math
import numpy as np
import operator

Data = pd.read_csv('iris.data.learning', header=None)
LearningData = Data.values
Data = pd.read_csv('iris.data.test', header=None)
TestingData = Data.values

# utworzenie obiektu, który zawiera dane uczące, ale bez etykiet
def getDataWithoutLabels(Data):
    withoutLabels = []
    for x in Data:
        withoutLabels.append((x[0:4]))
    # print(withoutLabels)
    return withoutLabels

# okresla etykietę danej testowej wedlug etykiet sąsiadów (ze zbioru danych uczących)
def checkLabel(neighbours):
    labels = {}
    for i in range(len(neighbours)):
        label = neighbours[i][-1]
        if label in labels:
            labels[label] += 1
        else:
            labels[label] = 1
    # print(i, labels)
    sortedLabels = sorted(labels.__iter__(), key=operator.itemgetter(-1))
    return sortedLabels[0]

# okresla dystans miedzy dana uczącą a daną testową za pomocą metryki Euklidesowej
def distance(p1, p2):
    length = math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2)
                       + math.pow((p1[2] - p2[2]), 2) + math.pow((p1[3] - p2[3]), 2))
    return length


# klasa implementująca algorytm kNN
class kNN:

    # konstruktor określa k najbliższych sąsiadów i dane uczące  z etykietami
    def __init__(self, k, learning_data):
        self.k = k
        self.Data = learning_data

    # przyjmuje listę danych (bez etykiet) do klasyfikacji czyli przypisania im etykiet
    # zwraca listę rozpoznanych etykiet
    def predict(self, testing_data):
        labels = []
        for y in range(len(testing_data)):
            distances = []
            for x in range(len(self.Data)):
                point = self.Data[x]
                dist = distance(point, testing_data[y])
                distances.append((point, dist))
            distances.sort(key=operator.itemgetter(1))
            neighbours = []
            for i in range(self.k):
                neighbours.append(distances[i][0])
            labels.append(checkLabel(neighbours))
        # print(neighbours)
        return labels

    # przyjmującą listę z obiektami do klasyfikacji (bez etykiet) oraz listę etykiet
    # Zwraca współczynnik poprawnie rozpoznanych obiektów
    def score(self, testing_data, labels):
        matches = 0
        for i in range(len(testing_data)):
            Data = testing_data[i]
            if (Data[4] == labels[i]):
                matches += 1
        return (matches)


# utworzenie obiektu klasy kNN z 5 najbliższymi sąsiadami i danymi uczącymi
kNN = kNN(5, LearningData)

# utworzenie obiektu zawierającego dane testujące bez etykiet
TestingDataWithoutLabels = getDataWithoutLabels(TestingData)

# określenie etykiet do danych testujących
CheckedLabels = kNN.predict(TestingDataWithoutLabels)

accuracy = (kNN.score(TestingData, CheckedLabels) / len(TestingData)) * 100
print("Accuracy of our model is equal ", str(round(accuracy, 2)), "%")
