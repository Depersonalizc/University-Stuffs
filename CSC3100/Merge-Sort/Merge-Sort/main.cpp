#include <iostream>

using namespace std;

const uint32_t INF = 0xFFFFFFFF;

void merge(uint32_t* arr, int lo, int mid, int hi) {
    auto len = hi - lo + 1;
    uint32_t tmp[len+2]; // for two infinities

    int i = 0;
    for (; i <= mid - lo; i++) tmp[i] = arr[lo+i];
    tmp[i++] = INF;
    for (; i < len+1; i++) tmp[i] = arr[lo+i-1];
    tmp[i] = INF;

    // pointer within arr
    int p  = lo;
    // pointers within tmp
    int p1 = 0;
    int p2 = mid+2 - lo;
    while (true) {
        if (tmp[p1] < tmp[p2]) arr[p++] = tmp[p1++];
        else arr[p++] = tmp[p2++];
        if (p == lo + len) break;
    }
}

void mergeSort(uint32_t* arr, int lo, int hi) {
    if (lo < hi) {
        auto mid = (lo + hi) / 2;
        mergeSort(arr, lo, mid);
        mergeSort(arr, mid+1, hi);
        merge(arr, lo, mid, hi);
    }
}

int main()
{
    int N;
    cin >> N;
    uint32_t arr[N];
    for (int i = 0; i < N; i++) {
        cin >> arr[i];
    }
    mergeSort(arr, 0, N-1);
    for (int i = 0; i < N; i++) {
        cout << arr[i] << endl;
    }

//    uint32_t arr[] = {1,8,9,2,3};
//    merge(arr, 0, 2, 4);
////    for (int i = 0; i < 5; i++) {
////        cout << arr[i] << ", ";
////    }
}
