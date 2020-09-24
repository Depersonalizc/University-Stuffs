/*
 * File: q3.cpp
 * ----------------------------------
 * This program removes comments from a C++ code and outputs
 * the version without the comments under the same path.
 * There should NOT be any string containing comment symbols
 * (e.g. string foo = "// comments";)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include "q3.h"

using namespace std;

// Constants and Types
const string HEADER = "CSC3002 Assignment1 Question3: Remove Comments";
const string PROMPT_IN = "Enter abosulte input file path: ";
const string PROMPT_OUT = "The output file is: ";
const string ENDPRO = "End of Question3";

/*
 * Testing Function of Question 3
 * Prompts user for path of input file, removes comments from the file,
 * and saves under the same path.
 */
void q3() {
  string name_out = "output.txt", f_path;
  ifstream ifs;
  ofstream ofs;

  cout << HEADER << endl;
  while (1) {
  cout << PROMPT_IN << endl;
  cin >> f_path;

  try {
      string in_string = readFileFromPath(ifs, f_path);
      istringstream iss(in_string);
      int slash = f_path.find_last_of('\\');
      f_path = f_path.substr(0, slash + 1) + name_out;
      ofs.open(f_path);
      removeComments(iss, ofs);
      ofs.close();
      cout << PROMPT_OUT << f_path << endl;
      cout << ENDPRO << endl;
      return;
  } catch (const char* msg) {
      cerr << msg << endl;
      continue;
  }
 }

}

/*
 * Function: removeComments
 * Usage: removeComments(is, os);
 * ----------------------------------
 * Copies input stream to output stream with comments removed.
 */
void removeComments(istream &is, ostream &os) {
  char cur;
  bool after_slash = 0, after_ast = 0, mult_com = 0, sing_com = 0;

  while (is.get(cur)) {
    if (mult_com) {
      if (after_ast && cur == '/') {  // found "*/", multiline comment ends
        mult_com = 0;
        after_ast = 0;
      } else if (cur == '*')
        after_ast = 1;
    } else {
      if (after_slash) {  // has not been comment
        switch (cur) {
          case '/':  // found "//", single-line comment starts
            sing_com = 1;
            after_slash = 0;
            break;
          case '*':  // found "/*", multiline comment starts
            mult_com = 1;
            after_slash = 0;
            break;
          default:  // false alarm from slash
            os << '/' << cur;
            after_slash = 0;
        }
      } else if (sing_com && cur == '\n') {  // end of single-line comment
        sing_com = 0;
        os << '\n';
      } else if (cur == '/')
        after_slash = 1;
      else
        os << cur;
    }
  }
}

/*
 * Function: readFileFromPath
 * Usage: string file = readFileFromPath(ifs, "C:\users\usr\test.txt");
 * ----------------------------------
 * Reads the text through path into the input file stream,
 * returns whole text in a string
 */
string readFileFromPath(ifstream &ifs, string path) {
  ifs.open(path);
  if (ifs.fail()) throw "Error: Failed to read the file!";
  ostringstream tmp;
  tmp << ifs.rdbuf();
  ifs.close();
  return tmp.str();
}
