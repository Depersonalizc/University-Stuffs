import sys
sys.setrecursionlimit(20000)

n = 13
q = [-1] * n
flag = 1

def safe(row):
    for i in range(row):
        if (
            q[i] == q[row] or  # same col
            abs(q[i] - q[row]) == abs(i - row)  # same diag
            ):
            return False
    return True

def show(q):
    n = len(q)
    for i in range(n):
        print('|', end='')
        for j in range(n):
            print(
                'Q' if q[i] == j else ' ', end='|'
            )
        print()
    print()

def backTrack(row):
    global flag
    if row == n:
        #show(q)
        return
    if (row, q[row]) == (0, n - 1):  # last tile reached
        flag = 0
        return
    for col in range(q[row] + 1, n):
        q[row] = col
        if safe(row):
            return backTrack(row + 1)
    q[row] = -1
    return backTrack(row - 1)

def main():
    i = 0
    backTrack(0)
    while flag:
        backTrack(n - 1)
        i += 1
    print('Total solutions: {}'.format(i))

import cProfile
cProfile.run(
    'main()'
)