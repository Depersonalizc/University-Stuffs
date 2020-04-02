/*
 * File: intarray.cpp
 * ----------------------
 * Implementation of the IntArray class.
 */

#include <iostream>
#include <string>
#include "intarray.h"
using namespace std;

/* Method implementation */

/*
 * Constructor: IntArray
 * Usage: IntArray iarr;    // IntArray iarr(10)
 *        IntArray iarr(n);
 * ----------------------
 * Initializes a new IntArray of n (10 if unspecified) integers, all set at zero.
 */

IntArray::IntArray(int n){
    array = new int[n];
    capacity = n;
    for (int i = 0; i < n; i++) array[i] = 0;
}

IntArray::IntArray(){
    array = new int[INITIAL_SIZE];
    capacity = INITIAL_SIZE;
    array[0] = 0;
}

/*
 * Destructor: ~IntArray
 * Usage: (usually implicit)
 * -------------------------
 * Frees any heap storage associated with this IntArray.
 */

IntArray::~IntArray(){
    delete [] array;
};

/*
 * Method: size
 * Usage: int nElems = iarr.size();
 * --------------------------------
 * Returns the number of elements in the IntArray.
 */

int IntArray::size() const{
    return capacity;
};

/*
 * Method: get
 * Usage: int elem = iarr.get(k);
 * --------------------------------
 * Returns the k-th element of the IntArray.
 * If k is outside the vector bounds, throws error.
 */

int IntArray::get(int k) const {
    if (k < 0 || k > capacity - 1) throw out_of_range("Cannot get element. Index out of bound!");
    return array[k];
};

/*
 * Method: put
 * Usage: iarr.put(k, value);
 * --------------------------------
 * Puts integer value at position k of the IntArray.
 * If k is outside the vector bounds, throws error.
 */

void IntArray::put(int k, int value){
    if (k < 0 || k > capacity - 1) throw out_of_range("Cannot put element. Index out of bound!");
    array[k] = value;
};

/*
 * Operator: []
 * Usage: iarr[k];
 * -----------------
 * Returns the k-th element of the IntArray by reference.
 */

int & IntArray::operator[](int k){
    if (k < 0 || k > capacity - 1) throw out_of_range("Cannot access element. Index out of bound!");
    return array[k];
};

/*
 * Operator: =
 * Usage: dst = src;
 * -----------------
 * Assigns src to dst so that the two IntArrays are independent copies.
 */

IntArray & IntArray::operator=(const IntArray & src){
    if (this != &src) {
        deepCopy(src);
    }
    return *this;
};

/*
 * Implementation notes: deepCopy
 * ------------------------------
 * This method copies the data from the src parameter into the current
 * object.  All dynamic memory is reallocated to create a "deep copy"
 * in which the current object and the source object are independent.
 */

void IntArray::deepCopy(const IntArray & src) {
   array = new int[src.capacity];
   for (int i = 0; i < src.capacity; i++) {
      array[i] = src.array[i];
   }
   capacity = src.capacity;
}
