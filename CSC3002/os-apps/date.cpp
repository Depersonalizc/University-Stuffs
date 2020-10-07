#include <date.h>

void date()
{
    char buff[24];
    std::time_t now = std::time(nullptr);
    std::tm* tmptr =  std::localtime(&now);
    std::strftime(buff, 24, "%a %Y-%m-%d %H:%M:%S", tmptr);
    std::printf("%s\n", buff);
}
