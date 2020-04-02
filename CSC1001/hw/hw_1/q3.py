def get_digit(n):
    return (
        n if n < 10 
        else sum(int(dgt) for dgt in str(n))
    )


def sum_of_double_even_place(n):
    sum = 0
    for i in str(n)[-2::-2]:
        double = int(i) * 2
        sum += get_digit(double)
    return sum


def sum_of_odd_place(n):
    odgts = str(n)[-1::-2]
    return sum(int(odgt) for odgt in odgts)


def is_valid(n):
    s = sum_of_double_even_place(n) + sum_of_odd_place(n)
    return True if s % 10 == 0 else False


print(
    is_valid(348906457280005)
)