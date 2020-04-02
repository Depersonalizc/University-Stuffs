# !/bin/env python
# -*- coding:utf-8 -*-
import collections as coll

string = (
    '-2*t^(-3)+5*t+1-4*t^2'  # change the polynomial here
)                            # negative power should be in parenthesis


class Polynomial:
    def __init__(self, p_list, c_list, var):
        self.pnc = coll.OrderedDict(  # the powers and coefficients
            sorted(                   # of the polynomial
                dict(zip(p_list, c_list)).items(),
                reverse=True,
            )
        )
        self.varName = var  # name of variable

    def firstDerivative(self):
        new_pnc = self.pnc
        try:
            del new_pnc[0]
        except:
            pass

        new_p = [pwr - 1 for pwr in new_pnc]
        new_c = [new_pnc[pwr] * pwr for pwr in new_pnc]

        return Polynomial(new_p, new_c, self.varName)

    def poly2str(self):
        x = self.varName
        repls = (
                ('*{}^0'.format(x), ''),
                ('{}^1'.format(x), x)
        )

        s = '+'.join(
            '{}*{}^{}'.format(self.pnc[p], x, p)
            if p >= 0 else
            '{}*{}^({})'.format(self.pnc[p], x, p)
            for p in self.pnc
        )
        for r in (repls):
            s = s.replace(*r, 1)
        return s


def str2poly(string):
    c_list = []
    p_list = []

    # locate variable name
    for ch in string[::]:
        if ch.isalpha():
            var = ch
            break

    # locate terms with a negative power
    lb = rb = 0
    while True:
        lb = string.find('(')
        rb = string.find(')')
        if lb == -1:
            break

        p = int(string[lb + 1: rb])

        while string[lb] != '*':
            lb -= 1
        # found times sign
        times = lb

        while (
            lb != 0 or
            string[lb] not in ('+', '-')
        ):
            lb -= 1

        c = float(string[lb: times])
        if c == int(c):
            c = int(c)
        c_list.append(c)
        p_list.append(p)
        string = string[:lb] + string[rb + 1:]

    # locate all terms with a non-neg power
    terms = string.replace('-', '+-').split('+')
    for term in terms:
        try:
            c = float(term)  # term == c?
        except:
            pnc = term.split('*')  # find p and c
            try:
                c = float(pnc[0])
            except:
                pass
            else:
                p = (
                    int(pnc[1].split('^')[1])
                    if '^' in pnc[1]
                    else 1
                )
        else:
            p = 0  # term == c

        if c == int(c):
            c = int(c)

        c_list.append(c)
        p_list.append(p)

    return Polynomial(p_list, c_list, var)


def main():
    # shows the first derivative of a polynomial
    myPolynomial = str2poly(string)
    myDerivative = myPolynomial.firstDerivative()
    print(myDerivative.poly2str())


if __name__ == "__main__":
    main()
