def sq(a,b):
    if abs(int(a ** .5) - a ** .5) < 0.0000000001 and abs(int(b ** .5) - b ** .5) < 0.00000000001:
        return True
    else:
        return False

x, y = 3, 2
for i in range(100):
    if i != 0:
        x = 3 * x + 4 * y
        y = 2 * x + 3 * y
    print(x,y)
    print(sq(x,y))