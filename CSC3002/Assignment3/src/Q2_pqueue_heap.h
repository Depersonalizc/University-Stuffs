/*
 * File: pqueue.h
 * --------------
 * This file exports the PriorityQueue class, a collection in which values
 * are processed in priority order.
 */

#ifndef _pqueue_h
#define _pqueue_h

#include "vector.h"
#include "error.h"
#include <iostream>

/*
 * Class: PriorityQueue<ValueType>
 * -------------------------------
 * This class implements a priority queue using a heap in which values
 * are processed in order of priority.  As in conventional English usage,
 * lower priority numbers correspond to higher effective priorities, so
 * that a priority 1 item takes precedence over a priority 2 item.
 */

template <typename T>
class PriorityQueue {

public:

    /*
     * Constructor: PriorityQueue
     * Usage: PriorityQueue<ValueType> pq;
     * -----------------------------------
     * Initializes a new priority queue based on a Heap,
     * which is initially empty.
     */
    PriorityQueue() = default;

    /*
     * Destructor: ~PriorityQueue
     * --------------------------
     * Frees any heap storage associated with this priority queue.
     */
    ~PriorityQueue() = default;

    /*
     * Method: dequeue
     * Usage: ValueType first = pq.dequeue();
     * --------------------------------------
     * Removes and returns the highest priority value.  If multiple
     * entries in the queue have the same priority, those values are
     * dequeued in the same order in which they were enqueued.
     */
    T dequeue();

    /*
     * Method: enqueue
     * Usage: pq.enqueue(value, priority);
     * -----------------------------------
     * Adds <code>value</code> to the queue with the specified priority.
     * Lower priority numbers correspond to higher priorities, which
     * means that all priority 1 elements are dequeued before any
     * priority 2 elements.
     */
    void enqueue(const T& value, double priority);

    /*
     * Method: peek
     * Usage: ValueType first = pq.peek();
     * -----------------------------------
     * Returns the value of highest priority in the queue, without
     * removing it.
     */
    T peek() const;

    /*
     * Method: isEmpty
     * Usage: if (pq.isEmpty()) ...
     * ----------------------------
     * Returns <code>true</code> if the priority queue contains no elements.
     */
    bool isEmpty() const;

    /*
     * Method: size
     * Usage: int n = pq.size();
     * -------------------------
     * Returns the number of values in the priority queue.
     */
    int size() const;

    /* Private section */

    /**********************************************************************/
    /* Note: Everything below this point in the file is logically part    */
    /* of the implementation and should not be of interest to clients.    */
    /**********************************************************************/

private:

    // Node structure for the underlying heap
    struct HeapNode {
        T data;
        double priority;
    };

    Vector<HeapNode> heap;

    void swapNode(int indexA, int indexB);

    int leftChildIndex(int index) const;

    int rightChildIndex(int index) const;

    int parentIndex(int index) const;

    /* returns priority of node at index if index in range of heap,
       otherwise return INFINITY
    */
    double getPriority(int index) const;

};

template<typename T>
void PriorityQueue<T>::swapNode(int indexA, int indexB) {
    HeapNode tmpA = heap[indexA];
    heap[indexA] = heap[indexB];
    heap[indexB] = tmpA;
}

template<typename T>
int PriorityQueue<T>::leftChildIndex(int index) const {
    return 2 * index + 1;
}

template<typename T>
int PriorityQueue<T>::rightChildIndex(int index) const {
    return 2 * index + 2;
}

template<typename T>
int PriorityQueue<T>::parentIndex(int index) const {
    return (index - 1) / 2;
}

template<typename T>
int PriorityQueue<T>::size() const {
    return heap.size();
}

template<typename T>
bool PriorityQueue<T>::isEmpty() const {
    return heap.size() == 0;
}

template<typename T>
double PriorityQueue<T>::getPriority(int index) const {
    return index < heap.size()? heap[index].priority : INFINITY;
}

template<typename T>
T PriorityQueue<T>::peek() const {
    if (heap.size() == 0) {
        error("PriorityQueue::peek: Trying to peek at an empty queue.");
    } else {
        return heap[0].data;
    }
}

template<typename T>
T PriorityQueue<T>::dequeue() {
    if (heap.size() == 0) {
        error("PriorityQueue::dequeue: Trying to dequeue an empty queue.");
    } else {
        // saves value for later return
        T value = heap[0].data;

        // move right-most child to root
        heap[0] = heap[heap.size() - 1];
        heap.remove(heap.size() - 1); // if right-most child and root coincide this also removes the root.

        // sifts down...
        int parent = 0, leftChild, rightChild, priorChild;
        while (true) {
            leftChild = leftChildIndex(parent);
            rightChild = rightChildIndex(parent);
            priorChild = getPriority(leftChild) < getPriority(rightChild)? leftChild : rightChild;
            if (getPriority(parent) <= getPriority(priorChild)) break; // parent prior than child(ren) -- heapified.
            swapNode(parent, priorChild);
            parent = priorChild;
        }

        return value;
    }
}

template<typename T>
void PriorityQueue<T>::enqueue(const T& value, double priority) {
    HeapNode node;
    node.data = value;
    node.priority = priority;
    heap.push_back(node);

    // sifts down...
    int child = heap.size() - 1, parent = parentIndex(child);
    while (getPriority(parent) > getPriority(child)) {
        swapNode(parent, child);
        child = parent;
    }
}

#endif
