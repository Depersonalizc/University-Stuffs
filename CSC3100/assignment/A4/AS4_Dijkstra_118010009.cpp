#define INF 2147483647

#include <cstdio>

typedef unsigned int u32;

/* Util */
template<typename T>
void swap(T & left, T & right) {
    T tmp(left);
    left = right;
    right = tmp;
}

/* Queue Class */
class MinPriorityQueue {

public:
    int  size;
    int* minHeap; // min heap of vnum, priority == d[vnum]
    int* loc; // loc[vnum] == index of vnum in the heap

    u32* prior; // priority (d) array, vnum indexed


    MinPriorityQueue() {}
    MinPriorityQueue(int n, u32* priority) {
        size = 0;
        minHeap = new int[n];
        loc = new int[n];
        prior = priority;
    }

    int parent(int i) {return (i - 1) / 2;}

    int leftChild(int i) {return (2 * i) + 1;}
    
    int rightChild(int i) {return (2 * i) + 2;}
    
    bool lessPrior(int i, int j) {return prior[minHeap[i]] < prior[minHeap[j]];}

    void swapElem(int i, int j) {
        swap(loc[minHeap[i]], loc[minHeap[j]]);
        swap(minHeap[i], minHeap[j]);
    }

    void minHeapify(int i) {
        auto l = leftChild(i);
        auto r = rightChild(i);
        auto least = (l < size && lessPrior(l, i))? l : i;
        if (r < size && lessPrior(r, least))
            least = r;
        if (least != i) {
            swapElem(i, least);
            minHeapify(least);
        }
    }

    int pop() {
        auto top = minHeap[0];
        swapElem(0, --size);
        minHeapify(0);
        return top;
    }

    void heapifyUp(int i) {
        while (i > 0 && lessPrior(i, parent(i))) {
            swapElem(i, parent(i));
            i = parent(i);
        }
    }
};

/* Graph Class */
class Graph {

struct Vertex {
   Vertex* next;
   int     vnum;
   u32     dist;

   Vertex(Vertex* next, int vnum, u32 dist) {
       this->next = next;
       this->vnum = vnum;
       this->dist = dist;
   }
};

public:
    int nVertices;
    Vertex**  adj;


    Graph() {}
    Graph(int n) {
        adj = new Vertex* [n];
        nVertices = n;
    }

    void insertEdge(int from, int to, u32 dist) {
        adj[from] = new Vertex(adj[from], to, dist);
    }

};

/* Algorithm */
class Dijkstra {
public:
    u32*  d;
    Graph G;
    MinPriorityQueue q;
    bool* visited;

    int source;

    Dijkstra() {
        int nVertices, nEdges;
        scanf("%d%d%d", &nVertices, &nEdges, &source);
        --source;

        // init G
        G = Graph(nVertices);
        for (int _ = 0; _ < nEdges; ++_) {
            int u, v, d;
            scanf("%d%d%u", &u, &v, &d);
            G.insertEdge(--u, --v, d);
        }

        // init d, q, visited
        d = new u32[nVertices];
        q = MinPriorityQueue(nVertices, d);
        visited = new bool[nVertices];

        for (int i = 0; i < nVertices; ++i) {
            d[i] = INF;
            q.minHeap[i] = q.loc[i] = i;
            visited[i] = false;
        }

        d[source] = 0;
        q.swapElem(0, source);
        q.size = nVertices;
    }

    void printD() {
        for (int i = 0; i < G.nVertices; ++i)
            printf("%d\n", d[i] < INF? d[i] : -1);
    }

    void findMinPath() {
        while (q.size > 0) {
            auto from = q.pop();
            for (auto v = G.adj[from]; v; v = v->next) {
                auto to = v->vnum;
                if (!visited[to] && d[to] > d[from] + v->dist) {
                    d[to] = d[from] + v->dist;
                    q.heapifyUp(q.loc[to]);
                }
            }
            visited[from] = true;
        }
    }
};


int main() {

    auto dijkstra = Dijkstra();
    dijkstra.findMinPath();
    dijkstra.printD();

    return 0;
}