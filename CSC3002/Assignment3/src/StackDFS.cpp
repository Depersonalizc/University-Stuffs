/*
 * File: StackDFS.cpp
 * ------------------
 * This program reimplements the depth-first search algorithm using an
 * explicit stack.
 */

#include "graphtypes.h"
#include "foreach.h"

void visit(Node* node);

void depthFirstSearch(Node* start, Set<Node*> & searched) {
    if (!searched.contains(start)) {
        visit(start);
        searched.insert(start);
        for (Arc* arc : start->arcs) {
            depthFirstSearch(arc->finish, searched);
        }
    }
}
