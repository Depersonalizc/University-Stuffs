#ifndef Q1_H
#define Q1_H
#include<string>

enum Month {
    JANUARY,FEBRUARY,MARCH,APRIL,MAY,JUNE,JULY,AUGUST,SEPTEMBER,OCTOBER,NOVEMBER,DECEMBER
};

// Function prototypes

/*
 * Function: daysInMonth
 * Usage: int days = dayInMonth(MARCH, 2020); // 31
 * ----------------------------------
 * Returns the numbers of days in a
 * given month of a given year.
 */
int daysInMonth(Month month, int year);

/*
 * Function: isLeapYear
 * Usage: bool isLeap = isLeapYear(2020); // true
 * ----------------------------------
 * Returns a boolean value indicating
 * whether a given year is a leap year.
 */
bool isLeapYear(int year);

/*
 * Function: monthToString
 * Usage: string monthString = monthToString(MARCH); // "March"
 * ----------------------------------
 * returns the corresponding string of a Month variable.
 */
std::string monthToString(Month month);

/*
 * Test function of Question 1
 */
void q1();

#endif // Q1_H
