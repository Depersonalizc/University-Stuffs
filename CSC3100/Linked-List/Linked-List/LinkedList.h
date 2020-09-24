#ifndef LINKEDLIST_H
#define LINKEDLIST_H

#include <iostream>

struct Node {
    int key;
    Node* next = nullptr;
};

class LinkedList {

private:

    Node* head;

    /* reverse start, start->next,... up until tail.
     * return a pointer to the tail.
     */
    Node* reversed(Node* start) {
        if (start->next == nullptr || start == nullptr) return start;
        Node* newHead = reversed(start->next);
        start->next->next = start;
        start->next = nullptr;
        return newHead;
    }


public:

    /* constructor */
    LinkedList() {
        head = nullptr;
    }

    /* reverse the whole list */
    void reverse() {
        head = reversed(head);
    }


    /* return whether list is empty */
    bool isEmpty() {
        return head == nullptr;
    }


    /* traverse through all nodes by printing the values */
    void traverse() {
        std::cout << '{';

        if (!this->isEmpty()) {
            auto current = head;
            while (current != nullptr) {
                std::cout << current->key << "->";
                current = current->next;
            }
        }
        std::cout << "NULL}" << std::endl;
    }


    /* insert Node with key k to the head of the list */
    void insert(int k) {
        Node* newNode = new Node;
        newNode->key = k;

        if (!this->isEmpty()) newNode->next = head;
        head = newNode;
    }


    /* return index of first node with key k
     * return -1 if failed
     */
    int find(int k) {
        if (!this->isEmpty()) {
            int idx = 0;
            Node* current = head;
            while (current != nullptr) {
                if (current->key == k) return idx;
                current = current->next;
                idx++;
            }
        }
        return -1;
    }


    /* remove head */
    void remove() {
        if (!this->isEmpty()) {
            Node* oldHead = head;
            head = head->next;
            delete oldHead;
        }
    }


    /* remove the node with first occurence of key k */
    void remove(int k) {

        /* if empty list */
        if (this->isEmpty()) return;

        Node* current = head;

        /* if head has key k */
        if (current->key == k){
            head = current->next;
            return;
        }

        /* if middle node has key k */
        while (current->next != nullptr) {
            if (current->next->key == k) {
                /* remove current->next */
                current->next = current->next->next;
                delete current->next;
                return;
            } else current = current->next;
        }
    }












};


#endif // LINKEDLIST_H
