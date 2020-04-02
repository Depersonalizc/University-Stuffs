# !/bin/env python
# -*- coding:utf-8 -*-


class Node:
    def __init__(self, e):
        self.val = e
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, node):
        ''' Appends a node to the tail of the SLL. '''
        if self.head is None:  # list is empty
            self.head = node
        else:
            self.tail.next = node
        self.tail = node


def len(pointer: Node):
    ''' Given a reference to the head of a singly linked list,
        returns the length of that list.
        '''
    try:
        pointer = pointer.next
    except:
        raise ValueError('Input should be a reference to a node.')
    else:
        if pointer is None:
            return 1
        return len(pointer) + 1


if __name__ == "__main__":
    L = 10
    SLL = LinkedList()

    for _ in range(L):
        SLL.append(Node(1))

    print(len(SLL.head))

