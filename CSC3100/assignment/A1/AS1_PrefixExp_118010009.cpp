#include <charconv>
#include <iostream>
#include <string>

using namespace std;

const long long MOD = 1000000007;

template <typename T>
class MiniStack {
private:
    static const int DEFAULT_SIZE = 10;
    int eff_size; // effective size
    T* data;
public:
    MiniStack(int size = DEFAULT_SIZE);

    bool is_empty();
    void push(T item);
    T pop();
    int size();
};

template <typename T>
MiniStack<T>::MiniStack(int size) {
    eff_size = 0;
    data = new T[size];
}

template <typename T>
bool MiniStack<T>::is_empty() {
    return eff_size == 0;
}

template <typename T>
void MiniStack<T>::push(T item) {
    data[eff_size++] = item;
}

template <typename T>
T MiniStack<T>::pop() {
    return data[--eff_size];
}

template <typename T>
int MiniStack<T>::size() {
    return eff_size;
}

/* Calculate and return the result of expression op1 [op] op2,
 * [op] must be one of +, -, *
 */
long long calculate(long long op1, long long op2, char op) {
    op1 %= MOD;
    op2 %= MOD;
    switch (op) {
    case '+': return (op1 + op2) % MOD;
    case '-': return (op1 - op2) % MOD;
    case '*': return (op1 * op2) % MOD;
    }
}

bool is_num(string & num_op) {
    return isdigit(num_op[0]);
}

void my_stoll(string & s, long long & i) {
   from_chars(s.data(), s.data() + s.size(), i);
}

int main()
{
    int N;
    cin >> N;

    MiniStack<string> inS(N);
    MiniStack<long long> outS(N/2 + 1);

    string num_op;
    for (int i = 0; i < N; i++) {
        cin >> num_op;
        inS.push(num_op);
    }

    long long op1, op2;
    for (int i = 0; i < N; i++) {
        num_op = inS.pop();
        /* num_op is a number */
        if (is_num(num_op)) {
            my_stoll(num_op, op1);
            outS.push(op1);
        }
        /* num_op is an operator */
        else {
            if (outS.is_empty()) {
                cout << "Invalid"; return 0;
            }
            op1 = outS.pop();
            if (outS.is_empty()) {
                cout << "Invalid"; return 0;
            }
            op2 = outS.pop();
            outS.push(
                calculate(op1, op2, num_op[0])
            );
        }
    }

    if (outS.size() > 1) {cout << "Invalid"; return 0;}
    cout << (outS.pop() + MOD) % MOD;

    return 0;
}