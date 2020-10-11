import csv
import os, sys
import matplotlib.pyplot as plt
import numpy as np

class Traffic:
    def __init__(self, file=None):
        self.xs = []    # traffic when clear
        self.n = 0
        self.x_mean = None
        self.x_S2 = None

        self.ys = []    # traffic when rain
        self.m = 0
        self.y_mean = None
        self.y_S2 = None

        self.Sp2 = None
        self.pooled_max_err = None
        self.welch_max_err = None

        if file:
            self.load_data(file)

    def load_data(self, file):
        self.__init__()
        with open(file, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)[1::]
            self.xs = [float(d[-1]) for d in data if d[5] == 'Clear']
            self.ys = [float(d[-1]) for d in data if d[5] == 'Rain']
            self.n, self.m = len(self.xs), len(self.ys)
            self.x_mean = sum(self.xs) / self.n
            self.y_mean = sum(self.ys) / self.m
            self.x_S2 = sum((x - self.x_mean) ** 2 for x in self.xs) / (self.n - 1)
            self.y_S2 = sum((y - self.y_mean) ** 2 for y in self.ys) / (self.m - 1)
            num = (self.n - 1) * self.x_S2 + (self.m - 1) * self.y_S2
            den = self.n + self.m - 2
            self.Sp2 = num / den
            self.pooled_max_err = 1.96 * self.Sp2 ** .5 * (1 / self.n + 1 / self.m) ** .5
            self.welch_max_err = 1.96 * (self.x_S2 / self.n + self.y_S2 / self.m) ** .5

    def plot(self):
        plt.scatter(self.xs, self.ys)
        if (self.a and self.b):
            X = np.arange(min(self.xs), max(self.xs), 0.01)
            Y = self.a + self.b * (X - self.x_mean)
            plt.plot(X, Y, 'r')
        plt.show()

    def print_stat(self):
        print("{:-^60}".format(" statistics"))
        print("{:<13}{:<13}{:<13}".format("", "x (clear)", "y (rain)"))
        print("{:<13}{:<13}{:<13}".format("size", self.n, self.m))
        print("{:<13}{:<13.2f}{:<13.2f}".format("mean", self.x_mean, self.y_mean))
        print("{:<13}{:<13.2f}{:<13.2f}".format("sample var", self.x_S2, self.y_S2))
        print("{:<13}{:<13.2f}({:<.2f})".format("Sp^2 (Sp)", self.Sp2, self.Sp2 ** .5))
        print("{:<13}{:<13.2f}".format("pooled error", self.pooled_max_err))
        print("{:<13}{:<13.2f}\n".format("welch error", self.welch_max_err))

traffic = Traffic(os.path.join(sys.path[0], 'traffic.csv'))
traffic.print_stat()