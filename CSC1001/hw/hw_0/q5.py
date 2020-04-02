def is_prime(x):

    if x < 2 or type(x) != int:
        return False

    for num in range(2, x - 1):
        if x % num == 0:
            return False

    return True


def primes_lt(x):
    # returns a list of primes that are less than x

    lst = []

    for num in range(2, x):

        if is_prime(num):
            lst.append(num)

    return lst


def show_primes():

    try:
        # prompts user for input (possibly float)
        N = int(input('Enter a number: '))

    except:
        print('Not a number! Please enter again.')

    else:

        idx = 0
        for prime in primes_lt(N):

            print(prime, end='\t')
            idx += 1

            if idx == 8:                        # start a new line once if 8 primes in a row

                print()
                idx = 0

        print('Done!')


while True:

    show_primes()
