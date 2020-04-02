/*
 * File: TestIntArray.cpp
 * ----------------------
 * Tests the IntArray class.
 */

#include "TestIntArray.h"
using namespace std;

/* Test Function of Q2 */

void testIntArray() {
   IntArray v1(5);
   test("v1.size()", v1.size(), 5);
   for (int i = 0; i < 5; i++) {
      test("v1[" + to_string(i) + "]", v1[i], 0);
      cout << "v1[" << i << "] = " << i << ";" << endl;
      v1[i] = i;
      test("v1[" + to_string(i) + "]", v1[i], i);
   }
   IntArray v2;
   v2 = v1;
   test("v2.size()", v2.size(), 5);
   for (int i = 0; i < 5; i++) {
      test("v2[" + to_string(i) + "]", v2[i], i);
   }
}

/*
 * Function: test
 * Usage: test(str, value, expected);
 * ----------------------------------
 * Generates a line of test output, making sure that the value
 * is equal to the expected parameter.
 */

void test(string str, int value, int expected) {
   cout << str << " -> " << value;
   if (value != expected) cout << " (should be " << expected << ")";
   cout << endl;
}
