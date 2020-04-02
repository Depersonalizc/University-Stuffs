/*
 * File: FindAreaCode.cpp
 * ----------------------
 * This program looks up a numeric area codes for the United States
 * and Canada.  The program works in both directions.  If the user
 * enters a number, the program prints out the state or province to
 * which that code is assigned.  If the user enters a name, it prints
 * out all the area codes assigned to that name.
 */

#include "FindAreaCode.h"

using namespace std;

const string PROMPT_IN = "Enter area code or state name: ";

/*
 * Function: readCodeFile
 * Usage: readCodeFile(filename, map);
 * -----------------------------------
 * Reads a data file representing area codes and locations into the map,
 * which must be declared by the client.  Each line must consist of the
 * area code, a hyphen, and the name of the state/province.
 */

void readCodeFile(string filename, map<int, string> & map) {
   int code;
   string line, state;
   ifstream ifs(filename);
   while (getline(ifs, line)) {
       code = stoi(line.substr(0, 3));
       state = line.substr(4);
       map.insert(pair<int, string>(code, state));
   }
   ifs.close();
}

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
               map< string,vector<int> > & stateToAreaCodeList) {
   for (auto p : areaCodeToState){
       auto areaCode = p.first;
       auto state = p.second;
       try {
          auto vec = stateToAreaCodeList.at(state);
          vec.push_back(areaCode);
          stateToAreaCodeList[state] = vec;
       } catch (...) {  // state not in map yet
           stateToAreaCodeList[state] = vector<int>{areaCode};
       }
   }
}

/* Test Functions */

/* Test Function of Q1.1 */

void findAreaCode() {
   map<int, string> areaCodeToState;
   string input;
   readCodeFile("AreaCodes.txt", areaCodeToState);
   while (1) {
       cout << PROMPT_IN;
       getline(cin, input);
       try {
           int code = stoi(input);
           cout << areaCodeToState[code] << endl;
       } catch (...) {
           for (auto p : areaCodeToState){
               if (p.second == input) cout << p.first << endl;
           }
       }
   }
}

/* Test Function of Q1.2 */

void findAreaCodeInvertMap() {
   map<int, string> areaCodeToState;
   map< string, vector<int> > stateToAreaCodes;
   string input;
   readCodeFile("AreaCodes.txt", areaCodeToState);
   invertMap(areaCodeToState, stateToAreaCodes);

   while (1) {
       cout << PROMPT_IN;
       getline(cin, input);
       try {
           int code = stoi(input);
           cout << areaCodeToState[code] << endl;
       } catch (...) {
           auto vec = stateToAreaCodes[input];
           for (auto code : vec){
               cout << code << endl;
           }
       }
   }
}
