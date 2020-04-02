# !/bin/env python
# -*- coding:utf-8 -*-

import queue


def getq(N, beg, aux, end):
    q = queue.Queue()
    q.put((N, beg, aux, end))
    t = 2 ** N - N - 1  # num of times to dequeue

    for _ in range(t):
        n, b, a, e = q.get()
        if n == 1:
            q.put((n, b, a, e))
        else:
            for cmd in (
                (n - 1, b, e, a),
                (1, b, a, e),
                (n - 1, a, b, e),
            ):
                q.put(cmd)
    return q


def HanoiTower(N: int):
    ''' Given a non-negative integer N, 
        returns a solution to 3-Hanoi with N plates.
        '''
    if type(N) is int and N > 0:
        q = getq(N, 'A', 'B', 'C')
        while not q.empty():
            cmd = q.get()
            print('{} --> {}'.format(cmd[1], cmd[3]))
    else:
        print('N should be a non-negative integer')


if __name__ == "__main__":
	HanoiTower(4)
