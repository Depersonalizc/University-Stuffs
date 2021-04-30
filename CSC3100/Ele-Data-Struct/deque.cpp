#define EMPTY -1
#include <iostream>

using namespace std;

class Deque {
private:
    int* Q;
    int n;
    int head;
    int tail;
public:
    Deque(int size);
    void enqueue_at_tail(int x);
    void enqueue_at_head(int x);
    int dequeue_at_tail();
    int dequeue_at_head();
    void printQ();
};

Deque::Deque(int size) {
    this->n = size;
    this->Q = new int[size];
    this->head = EMPTY;
    this->tail = 0;
    printQ();
}

void Deque::enqueue_at_tail(int x) {
    printQ();
    if (head == tail) {
        cout << "OVERFLOW\n";
        return;
    }
    if (head == EMPTY) {
        head = tail;
    }
    Q[tail] = x;
    tail = ++tail % n;
    printQ();
}

void Deque::enqueue_at_head(int x) {
    if (head == tail) {
        cout << "OVERFLOW\n";
        return;
    }
    if (head == EMPTY) {
        head = tail;
    }
    head = (--head + n) % n;
    Q[head] = x;
    printQ();
}

int Deque::dequeue_at_tail() {
    if (head == EMPTY) {
        cout << "UNDERFLOW\n";
        return -1;
    }
    tail = (--tail + n) % n;
    if (head == tail) {
        head = EMPTY;
    }
    printQ();
    return Q[tail];
}

int Deque::dequeue_at_head() {
    if (head == EMPTY) {
        cout << "UNDERFLOW\n";
        return -1;
    }
    int top = Q[head];
    head = ++head % n;
    if (head == tail) {
        head = EMPTY;
    }
    printQ();
    return top;
}

void Deque::printQ() {
    cout << "head:" << head << ' ';
    if (head != EMPTY) {
        int i = head;
        do {
            cout << Q[i] << ' ';
            i = ++i % n;
        } while (i != tail);
    }
    cout << "tail:" << tail << endl;
}


int main() {
    Deque q(3);
    q.enqueue_at_head(1);
    q.enqueue_at_tail(2);
    q.enqueue_at_head(3);
    cout << q.dequeue_at_head() << endl;
    cout << q.dequeue_at_tail() << endl;
    cout << q.dequeue_at_tail() << endl;
    cout << q.dequeue_at_head();
}
