/*
 * CS 106B/X Sample Project
 * last updated: 2018/09/19 by Marty Stepp
 *
 * This project helps test that your Qt Creator system is installed correctly.
 * Compile and run this program to see a console and a graphical window.
 * If you see these windows, your Qt Creator is installed correctly.
 */

#include "console.h"
#include "gwindow.h"
#include "FindAreaCode.h"
#include "TestIntArray.h"
#include "HFractal.h"

using namespace std;

const string Q1_1 = "Testing Q1.1: Area Codes...";
const string Q1_2 = "Testing Q1.2: Area Codes (Invert Map)...";
const string Q2   = "Testing Q2: IntArray...";
const string Q3   = "Testing Q3: H-Fractal...";

const string Q1_1_ = "End of Q1.1: Area Codes";
const string Q1_2_ = "End of Q1.2: Area Codes (Invert Map)";
const string Q2_   = "End of Q2: IntArray";
const string Q3_   = "End of Q3: H-Fractal";

/*
 * Main Function
 * -------------------------------------------
 * Uncomment the line of code according to which question you'd like to test.
 */

int main() {

    cout << Q1_1 << endl; findAreaCode(); cout << Q1_1_ << endl;

    //cout << Q1_2 << endl; findAreaCodeInvertMap(); cout << Q1_2_ << endl;

    //cout << Q2 << endl; testIntArray(); cout << Q2_ << endl;

    //cout << Q3 << endl; hFractal(); cout << Q3_ << endl;

    return 0;
}
