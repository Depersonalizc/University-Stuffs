def is_anagram(s1, s2):
    try:
        l1 = sorted(list(s1))
        l2 = sorted(list(s2))
    except:
        raise ValueError
    else:
        return True if l1 == l2 else False


def main():
    while True:

        s1 = input('Enter the first word: ')
        s2 = input('Enter the secend word: ')

        copula = (
            'are'
            if is_anagram(s1.lower(), s2.lower())
            else 'are not'
        )

        print(
            '"{}" and "{}" {} anagrams.'.format(
                s1, s2, copula
            )
        )


main()
