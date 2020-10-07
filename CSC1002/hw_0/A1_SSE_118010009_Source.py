# !/bin/env python
# -*- coding:utf-8 -*-

def swap(lst, i_1, i_2):
    lst[i_1], lst[i_2] = lst[i_2], lst[i_1]


def solvable(puzzle):

    inv = 0

    for i in range(7):
        for j in range(i + 1, 8):
            if puzzle[i] > puzzle[j]:           # count number of inversions
                inv += 1

    return inv % 2 == 0


def generate_pzl():                        # return a solvable puzzle as a list

    import random

    puzzle = list(range(1, 9))
    random.shuffle(puzzle)

    if not solvable(puzzle):                # interchange first tile with second
        swap(puzzle, 0, 1)                  # in case of unsolvable puzzle

    puzzle.insert(random.randint(0, 8), ' ')

    return puzzle


def print_pzl(puzzle):                          # print out puzzle from the list
    print()
    for line in zip(*[iter(puzzle)] * 3):       # print each three tiles
        for num in line:
            print(num, end=' ')
        print()


def solved(puzzle):

    solved = [
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    ]

    return puzzle == solved


def possible_dir(puzzle):
                                        # return a list of possible directions
    dirs = [
        'u, l'   , 'u, l, r'   ,    'u, r'
        'u, l, d', 'u, l, d, r', 'u, d, r'
        'l, d'   , 'l, d, r'   ,    'd, r'
    ]

    return dirs[puzzle.index(' ')]


def slide(puzzle, direction):               # slide tile in input direction
                                            # and return a new puzzle
    blk = puzzle.index(' ')
    toMove = blk + {'l': 1, 'r': -1, 'u': 3, 'd': -3}[direction]

    swap(puzzle, blk, toMove)

    return puzzle


def new_game():

    step = 0                        # game initialization
    puzzle = generate_pzl()

    while True:

        print_pzl(puzzle)
        pd = possible_dir(puzzle)

        print('Input sliding direction ({}) > '.format(pd), end='')

        while True:                        # obtain direction to slide from user

            try:
                direction = input()
                if direction in possible_dir(puzzle):
                    break

            except:
                continue

        # update puzzle
        puzzle = slide(puzzle, direction)
        step += 1                                                # count steps

        if solved(puzzle):                                       # check if solved

            print_pzl(puzzle)
            print('Puzzle solved in {} moves.                                                   \
                  Congratulations!'.format(step))

            break


if __name__ == '__main__':

    print('Welcome to 8-puzzle game...',
          'Press any key to start >', sep='\n', end='')

    input()                          # wait for user's response

    while True:

        new_game()

        print(
            'Do you want to start a new game (Y/N)? ', end=''
        )                                                   # end of game

        while True:

            newGame = input().lower()

            if newGame in ['y', 'n']:
                break

        if newGame == 'n':
            print('\nExiting...')
            break
