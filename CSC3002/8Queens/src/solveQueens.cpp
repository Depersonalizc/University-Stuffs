#include "solveQueens.h"

bool isValid(int nPlaced, int row, Grid<char> & board){
    // check same row
    for (int col = 0; col < nPlaced; col++){
        if (board[row][col] == 'Q') return 0;
    }

    // check up-left diagonal
    for (int i = 1; (row - i >= 0) && (nPlaced - i >= 0); i++){
        if ((board[row - i][nPlaced - i]) == 'Q') return 0;
    }

    // check down-left diagonal
    for (int i = 1; (row + i < board.numRows()) && (nPlaced - i >= 0); i++){
        if ((board[row + i][nPlaced - i]) == 'Q') return 0;
    }

    return 1;
}

void solveQueens(int nPlaced, Grid<char> & board, Vector<Grid<char>> & solutions){
        if (nPlaced == board.numCols()) solutions.add(board);
        else {
            for (int row = 0; row < board.numRows(); row++){
                if (isValid(nPlaced, row, board)) {
                    board[row][nPlaced] = 'Q';
                    solveQueens(nPlaced + 1, board, solutions);
                    // end of search for current board with queen placed at row
                    board[row][nPlaced] = ' ';
                }
            }
        }
}
