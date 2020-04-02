def isPrime(x): # determine primality

    if x < 2 or type(x) != int:
        return False
    for num in range(2, int(x / 2)):
        if x % num == 0:
            return False
    return True


def getNEmirps(n):

    x = 12
    i = 0

    while i != n:

        rev_x = int(str(x)[::-1])   # reversed number

        if x == rev_x:
            x += 1
            continue

        if isPrime(x) and isPrime(rev_x):
            yield x
            i += 1
        
        x += 1


def main():
    x = getNEmirps(100)

    for line in zip(*[x] * 10):
        for num in line:
            print(num, end='\t')
        print()


if __name__ == "__main__":
    main()
