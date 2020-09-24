/*
 * File: q1.cpp
 * ----------------------------------
 * This program is a simplified candelar.
 * It outputs the number of days of each month in a given year.
 */

#include <iostream>
#include <string>
#include "q1.h"
using namespace std;

// Constants and Types
const string HEADER = "CSC3002 Assignment1 Question1: Calender";
const string PROMPT = "Enter a year (e.g. 2020): ";
const string ENDPRO = "End of Question1";
const string ERROR_SIGN = "Error: The year should be a positive number.";
const string ERROR_YEAR = "Error: Please enter a valid year!";

/*
 * Test function of Question 1
 * Prompts for a year (A.D.) from the user (e.g. 2020,)
 * outputs the numbers of days of each month in that year
 */
void q1() {
  int year;
  string yr;

  cout << HEADER << endl;
  while (1) {
    cout << PROMPT;
    cin >> yr;
    try {
      year = stoi(yr);
      if (year > 0) break; // valid input
      cout << ERROR_SIGN << endl; // negative year, prompt again
    } catch (...) {
      cout << ERROR_YEAR << endl; // not an integer, prompt again
    }
  }

  cout << "Year " << year << endl;
  for (int month = JANUARY; month <= DECEMBER; month++) {
    cout << monthToString(Month(month)) << " has "
         << daysInMonth(Month(month), year) << " days." << '\n';
  }
  cout << ENDPRO << endl;
}

/*
 * Function: daysInMonth
 * Usage: int days = dayInMonth(MARCH, 2020); // 31
 * ----------------------------------
 * Returns the numbers of days in a
 * given month of a given year.
 */
int daysInMonth(Month month, int year) {
  switch (month) {
    case 1:
      return 28 + isLeapYear(year);

    case 0:
    case 2:
    case 4:
    case 6:
    case 7:
    case 9:
    case 11:
      return 31;
      break;

    default:
      return 30;
  }
}

/*
 * Function: isLeapYear
 * Usage: bool isLeap = isLeapYear(2020); // true
 * ----------------------------------
 * Returns a boolean value indicating
 * whether a given year is a leap year.
 */
bool isLeapYear(int year) {
  return (year % 400 == 0) || ((year % 4 == 0) && (year % 100 != 0));
}

/*
 * Function: monthToString
 * Usage: string monthString = monthToString(MARCH); // "March"
 * ----------------------------------
 * returns the corresponding string of a Month variable.
 */
std::string monthToString(Month month) {
  switch (month) {
    case 0:
      return "January";
      break;

    case 1:
      return "February";
      break;

    case 2:
      return "March";
      break;

    case 3:
      return "April";
      break;

    case 4:
      return "May";
      break;

    case 5:
      return "June";
      break;

    case 6:
      return "July";
      break;

    case 7:
      return "August";
      break;

    case 8:
      return "September";
      break;

    case 9:
      return "November";
      break;

    case 10:
      return "October";
      break;

    case 11:
      return "December";
      break;

    default:
      throw "Invalid input!";
  }
}
