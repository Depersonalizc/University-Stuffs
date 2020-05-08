/*
 * File: pqueue.h
 * --------------
 * This interface exports the PriorityQueue template class, which implements
 * a queue in which the elements are dequeued in priority order.
 */

#ifndef _pqueue_h
#define _pqueue_h

#include "error.h"

/*
 * Class: PriorityQueue<ValueType>
 * -------------------------------
 * This class implements a priority queue using a linked list in which
 * values are processed in order of priority.  As in conventional English
 * usage, lower priority numbers correspond to higher effective priorities,
 * so that a priority 1 item takes precedence over a priority 2 item.
 */

template <typename T>
class PriorityQueue {

public:

    /*
     * Constructor: PriorityQueue
     * Usage: PriorityQueue<ValueType> pq;
     * -----------------------------------
     * Initializes a new priority queue based on a linked list,
     * which is initially empty.
     */
    PriorityQueue();

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

    /* Node structure for the underlying linked list cell */
    struct Node{
        T data;
        Node* nextp;
        double priority;
    };

    /* Instance Varibales */
    Node* headp;
    Node* tailp;
    int count;

};

template <typename T>
PriorityQueue<T>::PriorityQueue() {
    headp = tailp = NULL;
    count = 0;
}

template <typename T>
int PriorityQueue<T>::size() const {
    return count;
}

template <typename T>
void PriorityQueue<T>::enqueue(const T& value, double priority) {
    Node* nodep = new Node;
    nodep->data = value;
    nodep->priority = priority;
    nodep->nextp = NULL;

    if (headp == NULL) { // if list is empty
        headp = nodep;
    } else {
        tailp->nextp = nodep;
    }
    tailp = nodep;
    ++count;
}

template <typename T>
T PriorityQueue<T>::peek() const {
    Node* scanp; // scannning pointer
    Node* maxPriorp; // points to node with max priority
    if (count == 0) {
        error("PriorityQueue::peek(): Trying to peek at an empty queue.");
    } else {
        scanp = maxPriorp = headp;
        while ((scanp = scanp->nextp)) {
            if (scanp->priority < maxPriorp->priority) {
                maxPriorp = scanp;
            }
        }
    }
    return maxPriorp->data;
}

template <typename T>
T PriorityQueue<T>::dequeue() {
    T value;
    Node* scanp; // scannning pointer
    Node* scanPrevp; // points to the node before the one scanning pointer points to
    Node* maxPriorp; // points to node with max priority
    Node* maxPriorPrevp; // points to node before the node with max priority

    if (count == 0) {
        error("PriorityQueue::dequeue(): Trying to dequeue an empty queue.");
    } else {
        scanp = maxPriorp = headp;
        scanPrevp = maxPriorPrevp = NULL;
        while ((scanp = scanp->nextp)){
            scanPrevp = (scanp == headp->nextp)? headp : scanPrevp->nextp;
            if (scanp->priority < maxPriorp->priority) {
                maxPriorp = scanp;
                maxPriorPrevp = scanPrevp;
            }
        }
    }
    value = maxPriorp->data; // stores node data for return

    if (maxPriorp == tailp) tailp = maxPriorPrevp; // updates tail

    if (maxPriorp == headp) headp = headp->nextp; // updates head
    else maxPriorPrevp->nextp = maxPriorp->nextp; // removes node from list

    delete maxPriorp; // frees heap memory associated with node
    --count;

    return value;
}

template <typename T>
bool PriorityQueue<T>::isEmpty() const {
    return count == 0;
}

#endif
