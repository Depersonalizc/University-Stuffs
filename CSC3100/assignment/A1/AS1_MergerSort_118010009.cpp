#include <stdio.h>

void merge(unsigned int* arr, int lo, int mid, int hi) {
    int len = hi - lo + 1;
    unsigned int tmp[len];
    for (int i = 0; i < len; i++) tmp[i] = arr[lo+i];

    // pointer within arr
    int p  = lo;
    // pointers within tmp
    int p1 = 0;
    int p2 = mid+1 - lo;
    while (true) {
        if (p1 > mid - lo) {
            while (p2 <= hi - lo) arr[p++] = tmp[p2++];
            break;
        }

        if (p2 > hi - lo) {
            while (p1 <= mid - lo) arr[p++] = tmp[p1++];
            break;
        }

        if (tmp[p1] < tmp[p2]) arr[p++] = tmp[p1++];
        else arr[p++] = tmp[p2++];
    }
}

void mergeSort(unsigned int* arr, int lo, int hi) {
    if (lo < hi) {
        int mid = (lo + hi) / 2;
        mergeSort(arr, lo, mid);
        mergeSort(arr, mid+1, hi);
        merge(arr, lo, mid, hi);
    }
}

int main()
{
    int N;
    scanf("%d", &N);
    unsigned int arr[N];
    for (int i = 0; i < N; i++) {
        scanf("%u", &arr[i]);
    }
    mergeSort(arr, 0, N-1);
    for (int i = 0; i < N; i++) {
        printf("%u\n", arr[i]);
    }
}