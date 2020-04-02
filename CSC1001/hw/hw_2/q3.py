# !/bin/env python
# -*- coding:utf-8 -*-
import random

size = 139
fishes = 20
bears = 5
rounds = 100


class Ecosystem:
    def __init__(self, size, fishes, bears):

        creatures = fishes + bears
        creatureLocs = random.sample(
            range(size), creatures
        )
        fishLoc = random.sample(creatureLocs, fishes)

        self.size = size
        self.river = [None] * size
        self.vacancy = []

        for loc in range(size):
            if loc in creatureLocs:
                self.river[loc] = (
                    Fish() if loc in fishLoc else Bear()
                )
            else:
                self.vacancy.append(loc)

    def simulation(self, N):
        for rd in range(1, N + 1):
            # round started
            print('round{}:'.format(rd))
            spHere = None

            for i in range(self.size):

                nxt = self.river[i]
                if spHere == nxt:  # skip if next sp just moved
                    continue

                spHere = nxt  # otherwise sp takes movement

                if spHere is not None:
                    if i == 0:
                        step = random.randint(0, 1)
                    elif i == self.size - 1:
                        step = random.randint(0, 1) - 1
                    else:
                        step = random.randint(0, 2) - 1

                    if step == 0:  # next sp if step == 0
                        continue

                    there = i + step
                    spThere = self.river[there]

                    if spThere is None:  # free to move to
                        self.river[i] = None
                        self.vacancy.append(i)
                        self.river[there] = spHere
                        self.vacancy.remove(there)
                    elif type(spHere) == type(spThere):  # multiply
                        try:
                            someWhere = random.choice(self.vacancy)
                        except:  # pass if no vacancy
                            pass
                        else:  # otherwise multiply somewhere
                            self.river[someWhere] = spHere.multiply()
                            self.vacancy.remove(someWhere)
                    else:  # different spp, hunting
                        self.river[i] = None
                        self.vacancy.append(i)
                        if (
                            type(spThere) is Fish
                        ):
                            self.river[there] = spHere
            # round completed
            self.show()

    def show(self):
        for sp in self.river:
            if sp is None:
                print('N', end='')
            else:
                print(
                    'F' if type(sp) is Fish else 'B', end=''
                )
        print()


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
    e.simulation(rounds)


if __name__ == "__main__":
    main()
