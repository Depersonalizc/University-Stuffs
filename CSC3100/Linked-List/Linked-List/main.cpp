#include <LinkedList.h>

using namespace std;

int main()
{
    LinkedList list;
    //cout << list.isEmpty();
    for (int i = 0; i < 100; i++) list.insert(i);

    cout << list.find(2) << endl;
    cout << list.find(1) << endl;
    cout << list.find(3) << endl;

    list.traverse();

    list.reverse();
    list.traverse();

    return 0;
}
