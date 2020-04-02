#ifndef TESTINTARRAY_H
#define TESTINTARRAY_H

#include <string>
#include <iostream>
#include "intarray.h"

using namespace std;

/* Function prototypes */

/* Test function of Q2 */

void testIntArray();

/*
 * Function: test
 * Usage: test(str, value, expected);
 * ----------------------------------
 * Generates a line of test output, making sure that the value
 * is equal to the expected parameter.
 */

void test(string str, int value, int expected);

#endif // TESTINTARRAY_H
