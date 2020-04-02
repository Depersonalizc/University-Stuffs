#ifndef FINDAREACODE_H
#define FINDAREACODE_H

#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

/* Function prototypes */

/*
 * Function: readCodeFile
 * Usage: readCodeFile(filename, map);
 * -----------------------------------
 * Reads a data file representing area codes and locations into the map,
 * which must be declared by the client.  Each line must consist of the
 * area code, a hyphen, and the name of the state/province.
 */

void readCodeFile(string filename, map<int, string> & map);

/*
 * Function: invertMap
 * Usage: invertMap(areaCodeToState, stateToAreaCodeList);
 * -------------------------------------------------------
 * Fills up the stateToAreaCodeList map by linking each state
 * to a vector of all the area codes that state contains.  The
 * stateToAreaCodeList map is created by the client and should
 * be empty when invertMap is called.  It is interesting to note
 * that the implementation doesn't need to check whether the state
 * is already in the stateToAreaCodeList.  If it isn't, selecting
 * the element creates a default value.
 */

void invertMap(map<int,string> & areaCodeToState,
               map< string,vector<int> > & stateToAreaCodeList);

/* Test functions */

/* Test function of Q1.1 */

void findAreaCode();

/* Test function of Q1.2 */

void findAreaCodeInvertMap();


#endif // FINDAREACODE_H
