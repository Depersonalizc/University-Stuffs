#include "console.h"
#include "solveQueens.h"
using namespace std;

const string GREETINGS = "This is an exercise program to solve for the 8-Queens Problem.";
const string GOODBYE = "End of the program. Goodbye!";
const uint Q_NUM = 8;
Grid<char> board(Q_NUM, Q_NUM, ' ');
Vector<Grid<char>> solutions;

int main(){
    cout << GREETINGS << endl;
    solveQueens(0, board, solutions);
    for (int i = 0; i < solutions.size(); i++){
        auto sol = solutions[i];
        cout << "Solution " << i + 1 << ":\n" << sol.toString2D() << endl;
    }
    cout << GOODBYE;
    return 0;
}
