lst = [1, 2, 33, 2, 9, 12, -1, 4, 7]

def bubbleSort(array):
    L = len(array)
    for l in range(L, 2, -1):
        for i in range(l - 1):
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]

def main():
    bubbleSort(lst)
    print(lst)

main()