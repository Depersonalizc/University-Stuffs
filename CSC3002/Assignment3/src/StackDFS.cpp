/*
 * File: StackDFS.cpp
 * ------------------
 * This program reimplements the depth-first search algorithm using an
 * explicit stack.
 */

#include "graphtypes.h"
#include "foreach.h"

void visit(Node* node);

/*
 * Function: DFS
 * Usage: DFS(node, searched);
 * ---------------------------------
 * Begins a DFS starting at the specified node that avoids
 * revisiting nodes that are in the searched set.
 */

void DFS(Node* start, Set<Node*> & searched) {
    if (!searched.contains(start)) {
        visit(start);
        searched.insert(start);
        for (Arc* arc : start->arcs) {
            DFS(arc->finish, searched);
        }
    }
}

/*
 * Function: depthFirstSearch
 * Usage: depthFirstSearch(node);
 * ---------------------------------
 * Wrapper function of DFS() to initialize a searched set.
 * Begins a DFS starting at the specified node.
 */

void depthFirstSearch(Node* start) {
    Set<Node*> searched;
    DFS(start, searched);
}
