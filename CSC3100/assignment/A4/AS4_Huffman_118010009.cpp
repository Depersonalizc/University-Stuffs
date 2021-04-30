#include <cstdio>
#include <string>
#include <iostream>

using namespace std;


class Node {

public:
    int freq;
    char c;
    Node *left, *right, *parent;


    Node() {}
    Node(int freq, char c, Node* left = nullptr, 
         Node* right = nullptr, Node* parent = nullptr) 
    {    
        this->freq = freq;
        this->c = c;
        this->left = left;
        this->right = right;
        this->parent = parent;
    }

    bool operator<(Node & RHS) {
        auto d = freq - RHS.freq;
        return d == 0? c < RHS.c : d < 0;
    }

    Node* operator+(Node & RHS) {
        auto node = new Node(freq + RHS.freq, c, this, &RHS);
        parent = RHS.parent = node;
        return node;
    }


};

/* Queue Class */
class MinPriorityQueue {

public:
    int    size;
    Node** minHeap;


    MinPriorityQueue(int n, Node** heap) {
        size = n;
        minHeap = heap;
        for (int i = n / 2 - 1; i >= 0; --i)
            minHeapify(i);
    }

    int parent(int i) {return (i - 1) / 2;}

    int leftChild(int i) {return (2 * i) + 1;}
    
    int rightChild(int i) {return (2 * i) + 2;}
    
    bool lessPrior(int i, int j) {return *(minHeap[i]) < *(minHeap[j]);}

    void minHeapify(int i) {
        auto l = leftChild(i);
        auto r = rightChild(i);
        auto least = (l < size && lessPrior(l, i))? l : i;
        if (r < size && lessPrior(r, least))
            least = r;
        if (least != i) {
            swap(minHeap[i], minHeap[least]);
            minHeapify(least);
        }
    }

    void heapifyUp(int i) {
        while (i > 0 && lessPrior(i, parent(i))) {
            swap(minHeap[i], minHeap[parent(i)]);
            i = parent(i);
        }
    }

    Node* pop() {
        Node* top = minHeap[0];
        swap(minHeap[0], minHeap[--size]);
        minHeapify(0);
        return top;
    }

    void push(Node* n) {
        minHeap[size] = n;
        heapifyUp(size++);
    }

};


/* Algorithm */
class Huffman {
public:
    Node* tree;
    Node* charset[256];

    void buildTree(string & str) {
        // count frequencies
        int distinct = 0;
        for (auto & c : str) {
            if (charset[int(c)])
                ++charset[int(c)]->freq;
            else {
                ++distinct;
                charset[int(c)] = new Node(1, c);
            }
        }
        // build Queue
        auto heap = new Node*[distinct];
        auto count = 0;
        for (int i = 0; count < distinct && i < 256; ++i) {
            if (charset[i])
                heap[count++] = charset[i];
        }
        MinPriorityQueue Q(distinct, heap);

        // build Tree
        for (int _ = 1; _ < distinct; ++_) {
            auto* left = Q.pop();
            auto* right = Q.pop();
            Q.push(*left + *right);
        }
        tree = Q.pop();
    }


    void encodeString(string & str) {
        for (auto & c : str) {
            // climb back up the tree
            string code;
            auto* ch = charset[int(c)];
            auto* pa = ch->parent;
            for (; pa; ch = ch->parent, pa = pa->parent)
                code += ch == pa->left? '0' : '1';

            // print code of char in reverse
            for (auto c = code.rbegin(); c != code.rend(); ++c)
                printf("%c", *c);
        }
    }

};

int main() {

    auto h = Huffman();

    string str;
    getline(cin, str);

    h.buildTree(str);
    h.encodeString(str);

    return 0;
}