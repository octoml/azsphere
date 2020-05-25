#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <time.h>

// #include <applibs/log.h>
// #include <applibs/gpio.h>
// #include "config.c"

// extern "C" {
// #include <stdbool.h>
// #include <errno.h>
// #include <string.h>
// #include <time.h>
#include "example.h"
// }

// typedef enum {
//     ExitCode_Success = 0,

//     ExitCode_Main_Led = 1
// } ExitCode;

int main(void)
{
    EXAMPLE test;
    test.Print();
    // Log_Debug("Starting CMake Hello World application...\n");

    // int fd = GPIO_OpenAsOutput(LED1[0], GPIO_OutputMode_PushPull, GPIO_Value_High);
    // if (fd < 0) {
    //     Log_Debug(
    //         "Error opening GPIO: %s (%d). Check that app_manifest.json includes the GPIO used.\n",
    //         strerror(errno), errno);
    //     return ExitCode_Main_Led;
    // }

    // const struct timespec sleepTime = {.tv_sec = 1, .tv_nsec = 0};
    // while (true) {
    //     GPIO_SetValue(fd, GPIO_Value_Low);
    //     nanosleep(&sleepTime, NULL);
    //     GPIO_SetValue(fd, GPIO_Value_High);
    //     nanosleep(&sleepTime, NULL);
    // }
}