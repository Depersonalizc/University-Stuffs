def sqrt(n, acc=0.000001):
    if n < 0:
        raise ValueError('Parameter n should not be negative.')
    if acc <= 0:
        raise ValueError('Parameter acc should be positive.')
    lastGuess = 1
    while True:
        nextGuess = (lastGuess + (n / lastGuess)) / 2
        if abs(nextGuess - lastGuess) < acc:
            return nextGuess
        else:
            lastGuess = nextGuess

print(
    sqrt(2)
)