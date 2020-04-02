def display_digits():

    try:

        # prompts user for input
        num = int(input('Enter an integer: '))

    except:

        print(
            'Invalid input. Please enter again.'
        )

    else:

        # interates thru digits of number and display
        for dgt in str(num):
            print(dgt)


while True:
    display_digits()
