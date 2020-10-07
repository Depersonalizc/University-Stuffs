#include <cal.h>

const char* DAYS_OF_WEEK = "Su Mo Tu We Th Fr Sa  ";
const int DAYS_IN_MONTH[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

void print_mid(char* s, int len, char const* end) {
    int lpad = (len - std::strlen(s)) / 2;
    int rpad = len - lpad - std::strlen(s);
    printf("%*s%s%*s%s", lpad, "", s, rpad, "", end);
}

bool is_leap_year(int year) {
  return (year % 400 == 0) || ((year % 4 == 0) && (year % 100 != 0));
}

int days_in_month(int mon, int year) {
    return DAYS_IN_MONTH[mon] + (mon == 1 && is_leap_year(year));
}

void cal()
{
    char mon_yr[14];
    std::time_t now = std::time(nullptr);
    std::tm* tmptr =  std::localtime(&now);
    std::strftime(mon_yr, 14, "%B %Y", tmptr);
    const int DAYS_IN_MON = days_in_month(tmptr->tm_mon, tmptr->tm_year);
    const int FIRST_WDAY = (tmptr->tm_wday - (tmptr->tm_mday % 7 - 1) + 7) % 7;
    int mday = 1;

    print_mid(mon_yr, 20);
    printf("%s\n", DAYS_OF_WEEK);
    printf("%*s", 3 * FIRST_WDAY, " ");

    while (mday <= DAYS_IN_MON) {
        printf("%2d ", mday);
        if ((mday + FIRST_WDAY) % 7 == 0) printf("\n");
        mday++;
    }
}
