#ifdef _WIN32
#include <windows.h>
#else

#include <ctime>

#endif

#include <cstdio>
#include <sys/time.h>
#include "get_timestamp.h"

#ifdef _WIN32
static LONGLONG gFrequency = -1;
#endif

void printTimeval(const timeval &stamp) {
    auto milli = stamp.tv_usec / 1000;
    char buffer [80];
    strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", localtime(&stamp.tv_sec));
    char currentTime[84] = "";
    sprintf(currentTime, "%s:%03ld", buffer, milli);
    printf("Time: %s \n", currentTime);
}

long long get_timestamp() {
    return get_timestamp(false);
}

long long get_timestamp(bool print) {
#ifdef _WIN32
    if ( gFrequency < 0 )
    {
       LARGE_INTEGER frequency;
       if (QueryPerformanceFrequency(&frequency))
          gFrequency = frequency.QuadPart;
       else
          gFrequency = 0;
    };

    LARGE_INTEGER stamp;
    if (gFrequency > 0 && QueryPerformanceCounter(&stamp))
       return (long long)(stamp.QuadPart/(gFrequency/1000000L));

    return (long long)GetTickCount64() * 1000L;
#else
    struct timeval stamp;
    gettimeofday(&stamp, nullptr);
    if (print) printTimeval(stamp);
    return stamp.tv_sec * 1000000 + stamp.tv_usec;
#endif
}

void print_timestamp() {
#ifdef _WIN32
    SYSTEMTIME timestamp = { 0 };
    GetLocalTime(&timestamp);
    printf("Time: %04u-%02u-%02u %02u:%02u:%02u.%03u\n", timestamp.wYear, timestamp.wMonth, timestamp.wDay, timestamp.wHour, timestamp.wMinute, timestamp.wSecond, timestamp.wMilliseconds);
#else
    struct timeval stamp;
    gettimeofday(&stamp, nullptr);
    printTimeval(stamp);
#endif
}
