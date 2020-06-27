/*
 * File: QueueBFS.cpp
 * ------------------
 * This program reimplements the breadth-first search algorithm using an
 * explicit queue.
 */

#include "graphtypes.h"
#include "foreach.h"
#include "queue.h"

void visit(Node* node);

/*
 * Function: breadthFirstSearch
 * Usage: breadthFirstSearch(node);
 * ---------------------------------
 * Begins a BFS starting at the specified node.
 */

void breadthFirstSearch(Node* start) {
    Set<Node*> searched; // contains nodes which have already been searched
    Queue<Node*> q {start}; // contains nodes whose neighbors are to be searched in order
    visit(start);
    searched.add(start);

    // searches neighbors of nodes in the queue
    Node *origin, *dest;
    while (!q.isEmpty()) {
        origin = q.dequeue();
        for (Arc* arc : origin->arcs) {
            dest = arc->finish;
            if (!searched.contains(dest)) {
                visit(dest);
                searched.add(dest);
                q.enqueue(dest);
            }
        }
    }
}
