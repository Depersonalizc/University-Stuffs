/*
 * File: PriorityQueueUnitTest.cpp
 * -------------------------------
 * This file implements a simple test of the PriorityQueue class.
 */

#include <cassert>
#include <iostream>
#include <string>
#include "Q1_pqueue_list.h"
//#include "Q2_pqueue_heap.h"
using namespace std;

class A {
public:
    int a;
    A() {a = 1;}
    virtual void disp() {cout << a << endl;}
};

class B : public A {
public:
    int b;
    B() {b = 2;}
    void disp() {cout << a << b << endl;}
};

class C : public B {
public:
    int c;
    C() {b = 1; c = 3;}
    void disp() {cout << a << b << c << endl;}
};

int main() {
    C oc;
    B* pb = &oc;
    pb->disp();
    return 0;
}
