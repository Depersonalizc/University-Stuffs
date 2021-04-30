#include <cstdio>
#include <cmath>
#include <queue>

using namespace std;

/* Utilities */

int max(int a, int b) {return a > b? a : b;}

/* Nodes of the AVL tree */

class Node {
public:
    int key;
    Node* left;
    Node* right;
    int height;

    void updateHeight() {
        height = max(left? left->height : -1,
                     right? right->height : -1) + 1;
    }

    Node* rightRotate() {
        auto* y = left;
        auto* B = y->right;
        
        left = B;
        y->right = this;

        updateHeight();
        y->updateHeight();

        return y;
    }

    Node* leftRotate() {
        auto* y = right;
        auto* B = y->left;

        right = B;
        y->left = this;

        updateHeight();
        y->updateHeight();

        return y;
    }
};

int getHeight(Node* n) {
    return n? n->height : -1;
}

int getBalance(Node* n) {
    return n? getHeight(n->left) -
              getHeight(n->right) : 0;
}

Node* newNode(int key, Node* left=nullptr, 
              Node* right=nullptr, int height=0)
{
    auto* n = new Node;
    n->key = key;
    n->left = left;
    n->right = right;
    n->height = height;
    return n;
}

Node* rotate(Node* n) {

    auto b = getBalance(n);

    if (b > 1) {
        if (getBalance(n->left) <= 0) // LR
            n->left = n->left->leftRotate();
        return n->rightRotate();
    }
    if (b < -1) {
        if (getBalance(n->right) >= 0) // RL
            n->right = n->right->rightRotate();
        return n->leftRotate();
    }

    return n;
}

void _inOrder(Node* n) {
    if (n) {
        _inOrder(n->left);
        printf("%d, ", n->key);
        _inOrder(n->right);
    }
}

void _preOrder(Node* n) {
    if (n) {
        printf("%d, ", n->key);
        _preOrder(n->left);
        _preOrder(n->right);
    }
}

Node* _minNode(Node* n) {
    for (; n->left; n = n->left) {}
    return n;
}

Node* _maxNode(Node* n) {
    for (; n->right; n = n->right) {}
    return n;
}

Node* _insertNode(Node* n, int key) {
    /* Regular BST insertion */
    if (!n)
        return newNode(key);

    if (key < n->key)
        n->left = _insertNode(n->left, key);
    else if (key > n->key)
        n->right = _insertNode(n->right, key);
    else
        return n; // no insertion made

    /* Rotation */
    n->updateHeight();
    return rotate(n);
}

Node* _deleteNode(Node* n, int key) {
    /* Regular BST delete */
    if (!n)
        return n;

    if (key < n->key)
        n->left = _deleteNode(n->left, key);

    else if (key > n->key)
        n->right = _deleteNode(n->right, key);
    
    else {
        if (!n->left || !n->right) { 
            // at least one child empty
            Node* tmp = n->left? n->left : n->right;
            if (tmp) {
                *n = *tmp;
                delete tmp;
            }
            else {
                delete n;
                return nullptr;
            }
        } else { 
            // both children non-empty
            auto* succ = _minNode(n->right);
            n->key = succ->key;
            n->right = _deleteNode(n->right, succ->key);
        }
    }

    /* Rotation */
    n->updateHeight();
    return rotate(n);
}


class AVL {
public:
    Node* root;

    AVL() {root = nullptr;}

    void insertNode(int key) {
        root = _insertNode(root, key);
        preOrder();
        inOrder();
    }

    void deleteNode(int key) {
        root = _deleteNode(root, key);
        preOrder();
        inOrder();
    }

    void inOrder() {
        printf("in: ");
        _inOrder(root);
        printf("\n");
    }

    void preOrder() {
        printf("\npre: ");
        _preOrder(root);
        printf("\n");
    }

    Node* minNode() {
        return _minNode(root);
    }

    // find a node with key closest to k
    Node* closest(int k) {
        auto d = [k] (Node* x) {return abs(x->key - k);};
        Node* node = root;
        Node* mostRecentLeftP = nullptr;
        Node* mostRecentRightP = nullptr;

        // empty tree
        if (!node) return nullptr;

        while (true) {
            if (node->key < k) {
                if (!node->right) {
                    // end of journey. compare node
                    // with its successor MRLP, if any
                    return (!mostRecentRightP ||
                            d(node) < d(mostRecentRightP))? 
                            node : mostRecentRightP;
                }
                // can still go right. update MRLP
                mostRecentLeftP = node;
                node = node->right;
            } else if (node->key > k) {
                if (!node->left) {
                    // end of journey, compare node
                    // with its predecessor MRLP, if any
                    return (!mostRecentLeftP || 
                            d(node) < d(mostRecentLeftP))? 
                            node : mostRecentLeftP;
                }
                // can still go left. update MRRP
                mostRecentRightP = node;
                node = node->left;
            } else {
                // equal key found
                return node;
            }
        }
    }

};


#include <stack>

void itw(Node* node) {
    stack<Node*> s;
    while (true) {
        if (node) {
            s.push(node);
            node = node->left;
        } else if (!s.empty()) {
            node = s.top();
            s.pop();
            printf("%d, ", node->key);
            node = node->right;
        } else return;
    }
}

void visit(Node* node) {printf("%d, ", node->key);}

void morris(Node* node) {
    while (node) {
        if (node->left) {
            // find predecessor node
            auto* pred = node->left;
            for (; pred->right && pred->right != node; pred = pred->right) {}

            if (pred->right == node) {
                // pred was bridged. restore right pointer
                // visit current node; discover right subtree
                pred->right = nullptr;
                visit(node);
                node = node->right;
            } else {
                // pred is not bridged.
                // bridge pred; discover left subtree
                pred->right = node;
                node = node->left;
            }
        } else {
            // no left subtree.
            // visit current node; discover right subtree
            visit(node);
            node = node->right;
        }
    }
}

Node* LCA(Node* T, Node* x, Node* y) {
    if (!T) return nullptr;
    if (T == x || T == y) return T;

    auto* left = LCA(T->left, x, y);
    auto* right = LCA(T->right, x, y);
    if (left && right) return T;
    return left? left : right;
}

Node* mirror(Node* node) {
    if (!node) return nullptr;
    auto* r = mirror(node->left);
    auto* l = mirror(node->right);
    node->left = l;
    node->right = r;
    return node;
}

void mirrorNonRecursive(Node* node) {
    queue<Node*> q({node});
    while (!q.empty()) {
        auto n = q.size();
        for (int _ = 0; _ < n; ++_) {
            node = q.front(); q.pop();
            if (node) {
                auto* l = node->left;
                auto* r = node->right;
                q.push(l);
                q.push(r);
                node->left = r;
                node->right = l;
            }
        }
    }
}

int MAX(Node* node) {
    if (!node) return -100000;
    for (; node->right; node = node->right) {}
    return node->key;
}

int MIN(Node* node) {
    if (!node) return 100000;
    for (; node->left; node = node->left) {}
    return node->key;
}

bool isBST(Node* T) {
    if (!T) return true;
    return (isBST(T->left) && MAX(T->left) < T->key
         && isBST(T->right) && MIN(T->right) > T->key);
}

int main() {
    auto t = AVL();

    // t.insertNode(9);
    t.insertNode(5);
    t.insertNode(10);
    t.insertNode(0);
    t.insertNode(6);
    t.insertNode(1);
    t.insertNode(-1);
    t.insertNode(2);
    t.insertNode(3);
    t.insertNode(4);

    printf("%d", isBST(t.root));

    // t.root = mirror(t.root);
    // t.preOrder();
    // t.inOrder();

    mirrorNonRecursive(t.root);
    t.preOrder();
    t.inOrder();

    printf("%d", isBST(t.root));

    // t.deleteNode(3);
    // t.deleteNode(5);
    // morris(t.root);
    // morris(t.root);

    // int k = 8;
    // printf("closest(%d) = %d\n", k, t.closest(k)->key);

    // auto lca = LCA(t.root, t.closest(0), t.closest(4))->key;
    // printf("LCA = %d\n", lca);

}