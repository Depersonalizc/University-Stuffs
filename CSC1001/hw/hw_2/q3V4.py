import random
#import numpy as np
#import matplotlib.pyplot as plt

size = 500
fishes = 10
bears = 4
N = 300


class Ecosystem:
    def __init__(self, size, fishes, bears):
        creatureLoc = random.sample(
            range(size), fishes + bears
        )
        fishLoc = random.sample(creatureLoc, fishes)
        self.river = [None] * size
        for loc in creatureLoc:
            self.river[loc] = (
                Fish() if loc in fishLoc else Bear()
            )
        self.size = size

    def simulation(self, N):
        #fishCounts = []
        #bearCounts = []

        for yr in range(N):
            #fishes = bears = 0
            sp = None

            for i in range(self.size):
                if sp == self.river[i]:  # exculeds case when sp moves right
                    continue
                sp = self.river[i]
                if sp != None:  # creature takes movement
                    '''if type(sp) == Fish:  # counts += 1
                        fishes += 1
                    else:
                        bears += 1'''

                    if i == 0:
                        step = random.randint(0, 1)
                    elif i == self.size - 1:
                        step = random.randint(0, 1) - 1
                    else:
                        step = random.randint(0, 2) - 1

                    if step == 0:
                        continue

                    there = i + step
                    spThere = self.river[there]

                    if spThere == None:  # empty there
                        self.river[i] = None  # creature moves
                        self.river[there] = sp  # to there
                    elif type(sp) == type(
                        spThere
                    ):  # same sp here and there
                        psb = [
                            idx  # all empty locations
                            for idx in range(self.size)
                            if self.river[idx] == None
                        ]
                        try:
                            someWhere = random.choice(psb)
                        except:  # pass if no vacancy
                            pass
                        else:  # otherwise multiply somewhere
                            self.river[
                                someWhere
                            ] = sp.multiply()
                    else:  # different spp
                        self.river[i] = None
                        if (
                            type(spThere) == Fish
                        ):  # if sp there is a fish
                            self.river[there] = sp  # to there
            # yr ends
            # fishCounts.append(fishes)
            # bearCounts.append(bears)

        # return fishCounts, bearCounts

    def show(self):
        for i in self.river:
            if i == None:
                print('-', end='')
            else:
                print(
                    'F' if type(i) == Fish else 'B', end=''
                )


class Bear:
    def __init__(self):
        pass

    def multiply(self):
        return Bear()


class Fish:
    def __init__(self):
        pass

    def multiply(self):
        return Fish()


def main():
    e = Ecosystem(size, fishes, bears)
    # data = \
    e.simulation(N)
    e.show()

    # show plot for animal numbers
    '''t = list(range(1, N + 1))
    plt.plot(t, data[0], t, data[1])
    plt.show()'''


if __name__ == "__main__":
    main()
