#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <time.h>

#include <applibs/log.h>
#include <applibs/gpio.h>
#include "config.c"

#include "example.h"
#include "kws.h"
#include "wav_data.h"

int16_t audio_buffer[16000]=WAVE_DATA;

int main(void)
{
    EXAMPLE test;
    test.Print();

    KWS kws(audio_buffer);
    kws.extract_features();

    int fd = GPIO_OpenAsOutput(LED1[0], GPIO_OutputMode_PushPull, GPIO_Value_High);
    if (fd < 0) {
        Log_Debug(
            "Error opening GPIO: %s (%d). Check that app_manifest.json includes the GPIO used.\n",
            strerror(errno), errno);
        return -1;
    }

    const struct timespec sleepTime = {.tv_sec = 1, .tv_nsec = 0};
    while (true) {
        GPIO_SetValue(fd, GPIO_Value_Low);
        nanosleep(&sleepTime, NULL);
        GPIO_SetValue(fd, GPIO_Value_High);
        nanosleep(&sleepTime, NULL);
    }
}