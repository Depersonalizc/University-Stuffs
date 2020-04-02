/*
 * File: intarray.h
 * ----------------------
 * Header file of the IntArray class.
 */

#ifndef INTARRAY_H
#define INTARRAY_H

class IntArray {

  public:

  /*
   * Constructor: IntArray
   * Usage: IntArray iarr;    // IntArray iarr(1)
   *        IntArray iarr(n);
   * ----------------------
   * Initializes a new IntArray of n (1 if unspecified) integers, all set at zero.
   */

   IntArray(int n);
   IntArray();

  /*
   * Destructor: ~IntArray
   * Usage: (usually implicit)
   * -------------------------
   * Frees any heap storage associated with this IntArray.
   */

  ~IntArray();

  /*
   * Method: size
   * Usage: int nElems = iarr.size();
   * --------------------------------
   * Returns the number of elements in the IntArray.
   */

  int size() const;

  /*
   * Method: get
   * Usage: int elem = iarr.get(k);
   * --------------------------------
   * Returns the k-th element of the IntArray.
   * If k is outside the vector bounds, throws error.
   */

  int get(int k) const;

  /*
   * Method: put
   * Usage: iarr.put(k, value);
   * --------------------------------
   * Puts integer value at position k of the IntArray.
   * If k is outside the vector bounds, throws error.
   */

  void put(int k, int value);

  /*
   * Operator: []
   * Usage: iarr[k];
   * -----------------
   * Returns the k-th element of the IntArray by reference.
   */

   int & operator[](int k);

   /*
    * Operator: =
    * Usage: dst = src;
    * -----------------
    * Assigns src to dst so that the two IntArrays are independent copies.
    */

    IntArray & operator=(const IntArray & src);



/* Private section */

private:

/* Private constants */

  static const int INITIAL_SIZE = 10;

/* Instance variables */

  int* array;          /* Dynamic array of integers   */
  int capacity;        /* Size of the array           */

/* Private method prototypes */

  void deepCopy(const IntArray & src);

};


#endif // INTARRAY_H
