T = [
    [33247, -48261, -40480, 108069],
    [-7248, 7866, 10022, -25693],
    [32166, -46710, -39150, 104520],
    [-13845, 17598, 18018, -47136],
]


def det(matrix):
    l = len(matrix)
    # for i in matrix:
    #     if len(i) != l:
    #         raise ValueError('Matrix not square')
    r0 = matrix[0]
    r1 = matrix[1]
    if l == 1:
        return r0[0]
    elif l == 2:
        return r0[0] * r1[1] - r0[1] * r1[0]
    else:
        ans = 0
        for col in range(l):
            sign = 2 * ((col + 1) % 2) - 1
            cols = list(range(col)) + list(range(col + 1, l))
            sub = [
                [row[t] for t in cols] for row in matrix[1:]
            ]
            ans += sign * r0[col] * det(sub)
        return ans
# T = [
#     [1, 0, 0, 0],
#     [5, 6, 0, 0],
#     [9, 10, 11, 0],
#     [13, 14, 15, 16],
# ]
def productFactor(x: int, n: int):
    if n == 1:
        return x
    for i in range(2, x):
        if x % i == 0:
            return i, productFactor(x // i, n - 1)
    return


# def arcdet(D, n):
#     D = 
#     T = [[0] * n for _ in range(n)]
#     for i in range(n):


print(productFactor(666, 4))
