def draw_table():

    while True:

        try:

            # prompts user for input
            N = int(input('Enter a positive integer: '))

            if N <= 0:

                print('Your integer must be positive!')

                # prompts again and print message if input not positive
                continue

        except:

            print('Not an integer. Please enter again.')

        else:

            print(
                # draw first line of the table
                '{}\t{}\t{}'.format('m', 'm+1', 'm**(m+1)')
            )

            # interates from 1 to N and prints output respectively
            for m in range(1, N+1):

                f_m = m + 1
                g_m = m ** (m+1)

                print(
                    '{}\t{}\t{}'.format(m, f_m, g_m)
                )


draw_table()
