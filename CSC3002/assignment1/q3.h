#ifndef Q3_H
#define Q3_H
#include <string>
// Function prototypes

/*
 * Function: readFileFromPath
 * Usage: string file = readFileFromPath(ifs, "C:\users\usr\test.txt");
 * ----------------------------------
 * Reads the text through path into the input file stream,
 * returns whole text in a string
 */
std::string readFileFromPath(std::ifstream & ifs, std::string path);

/*
 * Function: removeComments
 * Usage: removeComments(is, os);
 * ----------------------------------
 * Copies input stream to output stream with comments removed.
 */
void removeComments(std::istream & is, std::ostream & os);

/*
 * Testing Function of Question 3
 */
void q3();

#endif // Q3_H
