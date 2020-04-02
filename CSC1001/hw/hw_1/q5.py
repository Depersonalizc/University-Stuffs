lockers = [0] * 100
for stu in range(100):
    for i in range(stu, 100, stu + 1):
        lockers[i] = 0 if lockers[i] else 1
opens = ('L' + str(i + 1) for i in range(100) if lockers[i])

print('{} are open.'.format(', '.join(opens)))