import csv
import os, sys
import matplotlib.pyplot as plt
import numpy as np

class LinReg:
    def __init__(self, name=None, file=None, reg=True):
        self.name = name
        self.xs = []
        self.ys = []
        self.size = 0
        self.x_mean = None
        self.x_S2 = None
        self.y_mean = None
        self.y_S2 = None
        self.a = None
        self.b = None
        self.sigma2 = None
        self.SR2 = None
        self.a_max_err = None
        self.b_max_err = None

        if file:
            self.load_data(file)
            if reg: self.regress()

    def load_data(self, file):
        self.__init__(self.name)
        with open(os.path.join(file), newline='') as f:
            reader = csv.reader(f)
            data = list(reader)[1::]
            self.size = len(data)
            self.xs = [float(x) for [x, y] in data]
            self.ys = [float(y) for [x, y] in data]
            self.x_mean = sum(self.xs) / self.size
            self.y_mean = sum(self.ys) / self.size
            self.x_S2 = sum((x - self.x_mean) ** 2 for x in self.xs) / (self.size - 1)
            self.y_S2 = sum((y - self.y_mean) ** 2 for y in self.ys) / (self.size - 1)

    def regress(self, confidence=0.95):
        xy_sum = sum(x * y for x, y in zip(self.xs, self.ys))
        x2_sum = sum(x ** 2 for x in self.xs)
        y2_sum = sum(y ** 2 for y in self.ys)
        num = xy_sum - self.size * self.x_mean * self.y_mean
        den = x2_sum - self.size * self.x_mean ** 2
        x_SE = (self.size - 1) * self.x_S2
        t = 1.96 # should ideally be a t-quantile function, estimated by z_0.025

        self.a = self.y_mean
        self.b = num / den
        self.sigma2 = y2_sum / self.size - self.y_mean ** 2 - \
                      self.b * (xy_sum / self.size + self.x_mean * self.y_mean)
        self.SR2 = self.sigma2 * self.size / (self.size - 2)
        self.a_max_err = t * (self.SR2 / self.size) ** .5
        self.b_max_err = t * (self.SR2 / x_SE) ** .5

        self.print_stat()

    def plot(self):
        plt.scatter(self.xs, self.ys)
        if (self.a and self.b):
            X = np.arange(min(self.xs), max(self.xs), 0.01)
            Y = self.a + self.b * (X - self.x_mean)
            plt.plot(X, Y, 'r')
        plt.show()


    def print_stat(self):
        print("{} (n = {})".format(self.name, self.size))

        print("{:-^60}".format(" statistics"))
        print("{:<13}{:<13}{:<13}".format("", "x", "y"))
        print("{:<13}{:<13.2f}{:<13.2f}".format("mean", self.x_mean, self.y_mean))
        print("{:<13}{:<13.2f}{:<13.2f}\n".format("sample var", self.x_S2, self.y_S2))

        print("{:-^60}".format(" regression "))
        print("{:<13}{:<13}{:<13}{:<13}{:<13}".format("", "a^", "b^", "σ2^", "SR2"))
        print("{:<13}{:<13.2f}{:<13.2f}{:<13.2f}{:<13.2f}".format("estimator", self.a, self.b, self.sigma2, self.SR2))
        print("{:<13}{:<13.2f}{:<13.2f}{:<13.2}{:<13.2}\n".format("max error", self.a_max_err, self.b_max_err, "", ""))


dinosaur = LinReg("Dinosaur", os.path.join(sys.path[0], 'D.csv'))
star = LinReg("Star", os.path.join(sys.path[0], 'S.csv'))

dinosaur.plot()
star.plot()