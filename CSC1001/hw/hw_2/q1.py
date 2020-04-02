# !/bin/env python
# -*- coding:utf-8 -*-


class Flower:
    def __init__(self, name, patalNum, price):
        try:
            self.name = str(name)
            self.patalNum = int(patalNum)
            self.price = float(price)
        except:
            print('Invalid initial values!')

    # methods that set instance attributes
    def setName(self, newName):
        try:
            self.name = str(newName)
        except:
            print('Invalid name! (should be a string)')

    def setPatalNum(self, newNum):
        try:
            self.patalNum = int(newNum)
        except:
            print('Invalid number of patals! (should be an integer)')

    def setPrice(self, newPrice):
        try:
            self.price = float(newPrice)
        except:
            print('Invalid price! (should be a float)')

    # methods that retrieve instance attributes
    def showName(self):
        print(self.name)

    def showPatalNum(self):
        print(self.patalNum)

    def showPrice(self):
        print(self.price)
