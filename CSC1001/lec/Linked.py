class Node:
    def __init__(self, e):
        self.value = e
        self.prev = None
        self.nxt = None

class LinkedList:
    def __init__(self):
        self.header = Node(None)
        self.trailer = Node(None)

    def __len__(self):
        return 0

    def show(self):

    def insert(self, val, i):
        new = Node(val)
        count = 0
        now = self.header
        while count < i:
            now = now.nxt
            count += 1
        nxt = now.nxt
        nxt.prev = now.nxt = new
        new.prev, new.nxt = now, nxt

myList = LinkedList()

print(
    myList.header.value
)