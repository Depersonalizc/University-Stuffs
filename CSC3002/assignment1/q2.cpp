/*
 * File: q2.cpp
 * ----------------------------------
 * This program finds binding locations for two DNA strands
 * with the binding rules 'A' <-> 'T' and 'G' <-> 'C'.
 * Input DNA strands should only contain 'A', 'T', 'G', 'C'.
 */

#include "q2.h"
#include <string>
#include <iostream>
#include <vector>
using namespace std;

// Constants and Types
const string HEADER = "CSC3002 Assignment1 Question2: Find DNA Match";
const string PROMPT_DNA = "Enter a longer DNA strand (e.g. TAACGGTACGTC): ";
const string PROMPT_SNT = "Enter a shorter one (e.g. TGC): ";
const string ENDPRO = "End of Question2";

/*
 * Test function of Question 2
 * Prompts for two DNA strands, s1 (shorter), and s2 (longer),
 * Prints out ALL positions on s2 to which s1 can be attached.
 * Raises error info if inputs are invalid.
 */
void q2() {
  vector<int> found;
  string DNA_strand, snippet;
  int start = 0, loc = 0;

  cout << HEADER << endl;

  while (1) {
    cout << PROMPT_DNA;
    cin >> DNA_strand;
    cout << PROMPT_SNT;
    cin >> snippet;

    while (1) {
      try {
        loc = findDNAMatch(snippet, DNA_strand, start);
      } catch (const char* msg) {
        cerr << msg << endl;
        break;
      }

      // matching found
      if (loc >= 0) {
        found.push_back(loc);
        start = loc + 1;  // advance the starting base of searching
      }

      // all mathchings found, output the result.
      else {
        int size = found.size();

        if (size == 0)
          cout << "No matchings found!";
        else {
          cout << "Matching position(s) found at ";
          for (int i = 0; i < size - 1; i++) {
            cout << found[i] << ", ";
          }
          cout << found[size - 1] << ".\n";
        }

        cout << ENDPRO << endl;
        return;
      }
    }
  }
}

/*
 * Function: findDNAMatch
 * Usage: int loc = findDNAMatch("AT", "TATA"); // 0
 * ----------------------------------
 * Given two DNA strands (strings) s1, s2, and a starting position for s2,
 * returns the FIRST position on s2 after the starting position, to which
 * s1 (the shorter) can attach.
 */
int findDNAMatch(string s1, string s2, int start) {
  if (s1.length() > s2.length())
      throw "Error: s1 is longer than s2!";
  if (s2.find_first_not_of("ATCG", start) != s2.npos)
      throw "Error: Invalid strand!";

  // Converts s1 into its complementary strand
  for (int i = 0; i < int(s1.length()); i++) {
    switch (s1[i]) {
      case 'A':
        s1[i] = 'T';
        break;

      case 'T':
        s1[i] = 'A';
        break;

      case 'G':
        s1[i] = 'C';
        break;

      case 'C':
        s1[i] = 'G';
        break;

      default:
        throw "Error: Invalid strand!";
    }
  }

  return s2.find(s1, start);
}
