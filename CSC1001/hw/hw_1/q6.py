def toCheck(pos, n):  # generates idx to check for conflicts
    i, j = pos[0], pos[1]

    r1 = zip([i] * n, range(j))  # same row i
    r2 = zip([i] * n, range(j + 1, n))
    c1 = zip(range(i), [j] * n)  # same col j
    c2 = zip(range(i + 1, n), [j] * n)

    ld = i + j  # same left diag
    end = min(ld, n - 1)
    start = ld - end
    ld1 = zip(range(start, i), range(end, j, -1))
    ld2 = zip(
        range(i + 1, end + 1), range(j - 1, start - 1, -1)
    )

    rd = i - j  # same right diag
    start = max(0, rd)
    end = min(n - 1 + rd, n - 1)
    rd1 = zip(range(start, i), range(start - rd, j))
    rd2 = zip(
        range(i + 1, end + 1), range(j + 1, end - rd + 1)
    )
    for part in (r1, r2, c1, c2, ld1, ld2, rd1, rd2):
        for idx in part:
            yield idx

def n_queens_soln(
    n=8, oneSoln=0
):  # generates solutions for n-queen

    def safe(bd, idx):
        for i in bd[idx][1]:
            if bd[i][0]:
                return False
        return True

    def retrJ(p):  # retrieves previous j
        del p[-1]
        return [None] + p
   
    coords = ((x, y) for x in range(n) for y in range(n))
    board = {
        pos: [0, set(toCheck(pos, n))] for pos in coords
    }  # {pos: [queenOrNot, posToCheck]} 
    i = j = 0
    pvsJ = [None] * n

    while True:  # search in row i, col j
        if safe(board, (i, j)):
            board[(i, j)][0] = 1
            del pvsJ[0]
            pvsJ += [j]  # store j before going to the next row
            
            if i == n - 1:
                yield board  # yield if no next row
                if oneSoln:
                    return

                board[(i, j)][0] = 0  # remove the last queen
                pvsJ = retrJ(pvsJ)
                
                if j == n - 1:  # if last col
                    (i, j) = (i - 1, pvsJ[-1])  # backtrack to row n - 1
                    board[(i, j)][0] = 0
                    pvsJ = retrJ(pvsJ)
                j += 1
                continue

            else:
                (i, j) = (i + 1, 0)  # goto next row i + 1
                continue
        else:
            if j == n - 1:  # no sol found with prevj
                (i, j) = (i - 1, pvsJ[-1])
                pvsJ = retrJ(pvsJ)  # backtrack to row i - 1
                board[(i, j)][0] = 0  # remove the previous queen

                if j == n - 1:  # no sol found with pprevj
                    i, j = i - 1, pvsJ[-1]
                    pvsJ = retrJ(pvsJ)  # backtrack to row i - 2
                    try:
                        board[(i, j)][0] = 0
                    except:
                        return
            j += 1

def show(soln):
    n = int(len(soln) ** 0.5)
    board = (
        soln[key][0] for key in soln
        )
    for line in zip(*[board] * n):
        print('|', end='')
        for idx in line:
            print('Q' if idx else ' ', end='|')
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
    import cProfile
    cProfile.run('main()')
