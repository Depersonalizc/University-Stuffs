from math import sin, cos, tan


def numerical_int(f, a, b, n=10000):

    sum = 0

    for i in range(1, n+1):

        ptt = (b - a) / n
        arg = a + ptt * (i - .5)
        to_add = ptt * globals()[f](arg)            # calls function by str f

        sum += to_add

    return sum


while True:

    try:
        # prompts user for function name
        f = input('Function to integrate(sin, cos, or tan): ').lower()
        if f not in ['sin', 'cos', 'tan']:
            # prompts again if function not included
            print('function can only be sin, cos, or tan.')
            continue

        a = float(input('a = '))
        b = float(input('b = '))

        n = int(input('n = '))
        if n < 1:
            print('n should be a positive integer.')
            # prompts again if n not positive
            continue

    except:
        print('Invalid argument. Please try again')

    else:
        # displays the result
        rslt = numerical_int(f, a, b, n)
        print(
            '{}(x) integrated from {} to {} = {}'.format(f, a, b, rslt)
        )
