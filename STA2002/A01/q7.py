import csv
import os, sys

with open(os.path.join(sys.path[0], 'GMM.csv'), newline='') as f:
    reader = csv.reader(f)
    lst = list(reader)[1::]

n_0 = n_1 = s_0 = s_1 = q_0 = q_1 = 0
for r in lst:
    k, x = int(r[0]), float(r[1])
    if k:
        n_1 += 1
        s_1 += x
        q_1 += x**2
    else:
        n_0 += 1
        s_0 += x
        q_0 += x**2

pi_0  = n_0 / (n_0 + n_1)
mu_0  = s_0 / n_0
var_0 = q_0 / n_0 - (s_0 / n_0) ** 2
mu_1  = s_1 / n_1
var_1 = q_1 / n_1 - (s_1 / n_1) ** 2

print('pi_0  = ', pi_0, '\n',
        'mu_0  = ', mu_0, '\n',
        'var_0 = ', var_0, '\n',
        'mu_1  = ', mu_1, '\n', 
        'var_1 = ', var_1, sep='')