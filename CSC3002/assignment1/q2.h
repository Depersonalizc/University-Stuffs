#ifndef Q2_H
#define Q2_H
#include <string>
// Function prototypes

/*
 * Function: findDNAMatch
 * Usage: int loc = findDNAMatch("AT", "TATA"); // 0
 * ----------------------------------
 * Given two DNA strands (strings) s1, s2, and a starting position for s2,
 * returns the FIRST position on s2 after the starting position, to which
 * s1 (the shorter) can attach.
 */
int findDNAMatch(std::string s1, std::string s2, int start = 0);

/*
 * Test function of Question 2
 */
void q2();

#endif // Q2_H
