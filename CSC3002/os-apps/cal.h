#ifndef CAL_H
#define CAL_H

#include <cstdio>
#include <ctime>
#include <cstring>

void print_mid(char* s, int len, char const* end = "\n");

bool is_leap_year(int year);

int days_in_month(int mon, int year);

/*
 * Displays a calendar for the current month.
 */
void cal();

#endif // CAL_H
