#ifndef SOLVEQUEENS_H
#define SOLVEQUEENS_H

#include "grid.h"
#include "vector.h"

using namespace std;

bool isValid(int nPlaced, int row, Grid<char> & board);

void solveQueens(int nPlaced, Grid<char> & board, Vector<Grid<char>> & solutions);

#endif // SOLVEQUEENS_H
