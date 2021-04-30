#include <iostream>
#include <string>

using namespace std;

template <typename T>
class MiniStack {
private:
    static const int DEFAULT_SIZE = 10;
    int size;
    int eff_size; // effective size
    T* data;
public:
    MiniStack(int _size = DEFAULT_SIZE);

    void push(T item);
    T pop();
};

template <typename T>
MiniStack<T>::MiniStack(int _size) {
    size = _size;
    eff_size = 0;
    data = new T[size];
}

template <typename T>
void MiniStack<T>::push(T item) {
    data[eff_size++] = item;
}

template <typename T>
T MiniStack<T>::pop() {
    return data[--eff_size];
}

int main() {
    MiniStack<string> s;
    s.push(string("123"));
    s.push(string("jjj"));
    cout << s.pop() << s.pop();
}