lst = [3, 5, 4, 3, 2, 1, 0, 3, 5, 4, 3, 2, 1, 0, 3, 5, 4, 3, 2, 1,
       ]


def quickSort(arr, lo, hi):
    print(arr, lo, hi)
    if lo < hi:
        p = partition(arr, lo, hi)
        quickSort(arr, lo, p - 1)
        quickSort(arr, p + 1, hi)


def partition(arr, lo, hi):
    pivot = arr[lo]
    i, j = lo, hi
    while i != j:
        while True:
            i += 1
            if arr[i] < pivot or i == j:
                break
        while True:
            j -= 1
            if arr[j] > pivot or i == j:
                break
            # found arr[i] >= pivot >= arr[j]
        arr[i], arr[j] = arr[j], arr[i]
             # swap value
    # i == j
    arr[lo], arr[i] = arr[i], arr[lo]
    return i


quickSort(lst, 0, len(lst) - 1)
print(lst)
