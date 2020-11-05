import numpy as np

class NeuralNetwork():

    def __init__(self):
        self.wts1 = np.array([[0.2, -0.3],
                              [0.4, 0.1],
                              [-0.5, 0.2]])
        self.wts2 = np.array([[-0.3], [-0.2]])
        self.bias1 = np.array([[-0.4, 0.2]])
        self.bias2 = np.array([[0.1]])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sder(self, x):
        return x * (1 - x)

    def activate(self, x, bias, wts):
        x = x.astype(float)
        find = np.add(np.dot(x, wts), bias)
        return self.sigmoid(find)

    def printmatrix(self, matrix):
        r, c = matrix.shape
        for i in range(r):
            for j in range(c):
                print(round(matrix[i][j], 3), end=" ")

    def train(self, x, label, lr):
        for iteration in range(40000):
            o1 = self.activate(x, self.bias1, self.wts1)
            o2 = self.activate(o1, self.bias2, self.wts2)
            error1 = self.sder(o2) * (label - o2)
            error2 = np.multiply((self.sder(o1) * error1).T, self.wts2)
            self.wts1 += lr * np.multiply(x.T, error2.T)
            self.wts2 += lr * np.multiply(o1.T, error1)
            self.bias1 += np.multiply(lr, error2.T)
            self.bias2 += lr * error1


if __name__ == "__main__":
    ann = NeuralNetwork()

    print("INITIAL WEIGHTS (L->R)")
    ann.printmatrix(ann.wts1)
    ann.printmatrix(ann.wts2)
    print("\nINITIAL BIAS (L->R)")
    ann.printmatrix(ann.bias1)
    ann.printmatrix(ann.bias2)

    x = np.array([[1, 0, 1]])
    label = np.array([[1]])
    lr = 0.9

    ann.train(x, label, lr)

    print("\nADJUSTED WEIGHTS (L<-R)")
    ann.printmatrix(ann.wts2)
    ann.printmatrix(ann.wts1)
    print("\nADJUSTED BIAS (L<-R)")
    ann.printmatrix(ann.bias2)
    ann.printmatrix(ann.bias1)

    print("\n\nENTER INPUT TUPLE FOR PREDICTION: ")
    i1 = int(input("Feature-1: "))
    i2 = int(input("Feature-2: "))
    i3 = int(input("Feature-3: "))

    o1 = ann.activate(np.array([i1, i2, i3]), ann.bias1, ann.wts1)
    predict = ann.activate(o1, ann.bias2, ann.wts2)

    print("\n\nPREDICTED VALUE FOR TUPLE [%d %d %d] " % (i1, i2, i3))
    ann.printmatrix(predict)