#include <iostream>
#include <stack>
#include <queue>
// #include <unordered_map>

using namespace std;

struct Node {
    char val;
    Node* left;
    Node* right;
    Node(char val, Node* left = nullptr, Node* right = nullptr) {
        this->val = val;
        this->left = left;
        this->right = right;
    }
};


// void preorder(Node* x) {
//     if (x) {
//         cout << x->val;
//         preorder(x->left);
//         preorder(x->right);
//     }
// }


void preorder(Node* x) {
    auto s = stack<Node*>({x});
    while (!s.empty()) {
        auto tmp = s.top();
        s.pop();
        if (tmp) {
            cout << tmp->val;
            s.push(tmp->right);
            s.push(tmp->left);
        }
    }
}

// void inorder(Node* x) {
//     if (x) {
//         inorder(x->left);
//         cout << x->val << endl;
//         inorder(x->right);
//     }
// }

void inorder(Node* x) {
    // unordered_map<Node*, int> visited;
    auto s = stack<Node*>({x});
    while (!s.empty()) {
        auto tmp = s.top();
        s.pop();
        if (tmp) {
            // not visited

            if (tmp->right)
                s.push(tmp->right);

            s.push(tmp);
            s.push(nullptr);

            if (tmp->left)
                s.push(tmp->left);
        } else {
            // visited

            cout << s.top()->val;
            s.pop();
        }
    }
}

// void postorder(Node* x) {
//     if (x) {
//         postorder(x->left);
//         postorder(x->right);
//         cout << x->val << endl;
//     }
// }

void postorder(Node* x) {
    // unordered_map<Node*, int> visited;
    auto s = stack<Node*>({x});
    while (!s.empty()) {
        auto tmp = s.top();
        s.pop();
        if (tmp) {
            // not visited
            s.push(tmp);
            s.push(nullptr);

            if (tmp->right)
                s.push(tmp->right);

            if (tmp->left)
                s.push(tmp->left);

        } else {
            // visited
            cout << s.top()->val;
            s.pop();
        }
    }
}

int bfs(Node* r) {
    auto q = queue<Node*>({r});
    int level = 0;
    while (int size = q.size()) {
        cout << "Level " << level++ << ':' << endl;
        for (int _ = 0; _ < size; ++_) {
            auto tmp = q.front();
            q.pop();
            cout << tmp->val;
            if (tmp->left)
                q.push(tmp->left);
            if (tmp->right)
                q.push(tmp->right);
        }
        cout << endl;
    }
    return --level;
}

// find LCA of x and y in tree rooted at r.
Node* findLCA(Node* r, Node* x, Node* y) {
    if (r == x || r == y || !r)
        return r;
    auto left = findLCA(r->left, x, y);
    auto right = findLCA(r->right, x, y);
    if (left && right) return r;
    return left? left : right? right : nullptr;
}

void printLCA(Node* r, Node* x, Node* y) {
    auto LCA = findLCA(r, x, y);
    if (LCA)
        cout << "LCA: " << LCA->val;
    else
        cout << "Both nodes are outside the tree!";
    cout << endl;
}


int main() {
    // Node a('A'), b('B'), c('C'), d('D');
    // Node minus('-', &a, &b);
    // Node mult('*', &minus, &c);
    // Node plus('+', &mult, &d);

    // cout << bfs(&plus);

    Node d('D'), f('F'), g('G'), h('H'), 
         e('E', &g, &h), b('B', &d, &e),
         c('C', &f    ), a('A', &b, &c);

    Node out('O');

    // cout << bfs(&a);

    // printLCA(&a, &d, &h);
    // printLCA(&a, &b, &h);
    // printLCA(&a, &b, &f);
    printLCA(&a, &out, &out);


    return 0;
}