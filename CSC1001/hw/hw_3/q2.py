# !/bin/env python
# -*- coding:utf-8 -*-

from random import randint


class Node:
    def __init__(self, e):
        self.val = e
        self.next = None

    def track(self):
        rslt = LinkedList()
        crt = self
        while crt is not None:
            rslt.append(crt)
            crt = crt.next
        return rslt

    def copy(self):
        return Node(self.val)


class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def __len__(self):
        return self.size

    def __repr__(self):
        return '->'.join((str(n.val) for n in self._iter()))

    def _iter(self):
        n = self.head
        while n is not None:
            yield n
            n = n.next

    def append(self, node):
        ''' Appends a node to the tail of the SLL. '''
        if self.head is None:  # list is empty
            self.head = node
        else:
            self.tail.next = node
        self.tail = node
        self.size += 1

    def show(self):
        ''' Prints out all nodes of the SLL, head to tail. '''
        print(self.__repr__())

    def qsorted(self):
        ''' Returns a sorted version of the SLL
            between low and high in ascending order.
            '''

        def concat(*args):
            # Returns a concatenation of given SLL's
            rslt = LinkedList()
            for l in args:
                for n in l._iter():
                    rslt.append(n)
            return rslt

        # sorted by default
        if len(self) <= 1:
            return self

        # Lomuto partition: high as pivot
        pivot = self.tail

        # traverses the whole list except for pivot
        leftSub = LinkedList()
        rightSub = LinkedList()
        for n in self._iter():
            if n is not pivot:
                if n.val > pivot.val:
                    rightSub.append(Node(n.val))
                else:
                    leftSub.append(Node(n.val))

        # recursively sorts two sub-lists
        leftSub = leftSub.qsorted()
        rightSub = rightSub.qsorted()

        # concatenates sorted sub-lists with pivot
        p = LinkedList()
        p.append(pivot.copy())
        return concat(leftSub, p, rightSub)


def qsort(head: Node):
    ''' Given a reference to the head of an SLL, 
        returns a reference to the head of a sorted version.
        '''
    return head.track().qsorted().head


def main():
    test = [
        [1, 2, 3, 4, 5],
        [2, -9, 2, 0, -1],
        [11, 423, -5, 643, 22, 1],
    ]
    LL = LinkedList()

    for t in test:
        for i in t:
            LL.append(Node(i))

        sortedHead = qsort(LL.head)
        sortedLL = sortedHead.track()

        print('Before:')
        LL.show()
        print('After:')
        sortedLL.show()
        print()


if __name__ == "__main__":
    main()
