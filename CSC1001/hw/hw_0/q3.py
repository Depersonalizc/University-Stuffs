def least_int(m):

    # converts m into int to avoid possible float-int comparison
    m = int(m)
    n = 0                               # initializes n

    while n**2 <= m:                    # iterates until n^2 > m
        n += 1                          # each time adding 1 to n

    return n


while True:

    try:
        # prompts user for input (possibly float)
        user_num = float(input('Enter a number: '))

    except:
        print('Invalid input. Please enter again.')

    else:
        # converts float into int to avoid float-int comparison
        m = int(user_num)

        print(
            least_int(m)
        )
