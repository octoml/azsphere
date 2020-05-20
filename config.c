#pragma once
#include "Hardware/mt3620_rdb/inc/hw/sample_hardware.h"

#define AS_DEBUG                1
#define AS_NETWORKING           0
#define AS_ETHERNET_DEBUG       0


static int LED1[] = {MT3620_RDB_LED1_RED, MT3620_RDB_LED1_GREEN, MT3620_RDB_LED1_BLUE};
static int LED2[] = {MT3620_RDB_LED2_RED, MT3620_RDB_LED2_GREEN, MT3620_RDB_LED2_BLUE};
static int LED3[] = {MT3620_RDB_LED3_RED, MT3620_RDB_LED3_GREEN, MT3620_RDB_LED3_BLUE};
static int LED4[] = {MT3620_RDB_LED4_RED, MT3620_RDB_LED4_GREEN, MT3620_RDB_LED4_BLUE};

// MT3620 RDB: LED 1 (red channel)
// #define SAMPLE_LED MT3620_RDB_LED1_RED