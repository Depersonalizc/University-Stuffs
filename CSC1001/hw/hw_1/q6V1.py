def n_queens_soln(n=8, oneSoln=0):  # generates solutions for n-queen    
    queens = [None] * n
    i = 0

    def safe(queens, idx):
        for i in range(n):
            q = queens[i]
            if q != None:
                if q[1] == idx[1] or abs(q[1] - idx[1]) == abs(i - idx[0]):
                    return False
        return True

    while True:

        if i == n:
            #show(queens)
            yield queens
            if oneSoln:
                return  # stop searching if only one solution wanted

            i -= 1  # continue otherwise
            j = queens[i][1]

            if j != n - 1:
                queens[i] = (None, j)
            else:
                queens[i] = None
                i -= 1
                queens[i] = (None, queens[i][1])

        j0 = (
            0 if queens[i] == None else (queens[i][1] + 1)
        )  # col to start for search in ith row

        for j in range(j0, n):
            idx = (i, j)

            if safe(queens, idx):
                queens[i] = idx  # place queen if col is safe
                i += 1  # goto next row
                break
            else:  # if col not safe
                if j != n - 1:
                    continue  # check nxt col if not last row

                queens[i] = None  # otherwise no sol, remove the queen
                i -= 1  # backtrack to previous row

                if (
                    queens[i][1] == n - 1
                ):              # if queen in previous row is in last col
                    if i == 0:
                        return  # search stopped if queen in first row
                    queens[i] = None  # otherwise remove the queen
                    i -= 1  # and backtrack to previous row
                    queens[i] = (None, queens[i][1])
                    break
                else:
                    queens[i] = (
                        None,
                        queens[i][1],
                    )  # remove the queen


def show(soln):
    n = len(soln)
    board = ((x, y) for x in range(n) for y in range(n))

    for line in zip(*[board] * n):
        print('|', end='')
        for idx in line:
            print('Q' if idx in soln else ' ', end='|')
        print()

def main():
    x = n_queens_soln(
        n=8, oneSoln=0  # change para oneSoln to 1
    )  # if only one solution is wanted
    i = 0
    for sol in x:
        i += 1
        '''print('Solution {}:'.format(i))
        show(sol)
        print()'''
    print('Done!' if i else 'No solution found!')

if __name__ == "__main__":
    #main()
    import cProfile
    cProfile.run('main()')