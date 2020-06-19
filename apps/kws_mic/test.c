#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "lib/VectorTable.h"
#include "lib/CPUFreq.h"
#include "lib/UART.h"
#include "lib/Print.h"
#include "lib/GPIO.h"
#include "lib/GPT.h"
#include "lib/logical-dpc.h"

#include "include/config.h"
#include "include/intercore.h"
#include "include/tvmruntime.h"
#include "build/test_params.bin.c"
#include "build/test_graph.json.c"

#define _POSIX_C_SOURCE 200809L

#define FREQUENCY	197600000
#define buttonPressCheckPeriodMs	10

UART *debug = NULL;
static GPT *button_timer = NULL;
static GPT *exec_timer = NULL;
static uint32_t exec_timer_freq = 32768;
bool record;
static const uint32_t buttonBGpio = 13;
static const ComponentId HLCompID = {.data1 = 0x1689d8b2,
                                    .data2 = 0xc835,
                                    .data3 = 0x2e27,
                                    .data4 = {0x27, 0xad, 0xe8, 0x94, 0xd6, 0xd1, 0x5f, 0xa9}};
static void HandleButtonTimerIrq(GPT *);
static void HandleButtonTimerIrqDeferred(void);

void App_Init(void) {
	record = false;
	VectorTableInit();
	CPUFreq_Set(FREQUENCY);
	debug = UART_Open(MT3620_UNIT_UART_DEBUG, 115200, UART_PARITY_NONE, 1, NULL);
	GPIO_ConfigurePinForInput(buttonBGpio);
}

static void HandleButtonTimerIrq(GPT *handle)
{
	(void)handle;
	static CallbackNode cbn = {.enqueued = false, .cb = HandleButtonTimerIrqDeferred};
	EnqueueDeferredProc(&cbn);
}

static void HandleButtonTimerIrqDeferred(void)
{
  // Assume initial state is high, i.e. button not pressed.
	static bool prevState = true;
	bool newState;
	GPIO_Read(buttonBGpio, &newState);

	if (newState != prevState) {
		bool pressed = !newState;
		if (pressed) {
			record = true;
		}
		prevState = newState;
	}
}

_Noreturn void RTCoreMain(void)
{
	//debugging
	//change b to false for debugging
	volatile bool b = true;
	uint16_t count = 0;
	while (!b) {
		count++;
	}
	App_Init();
	uint32_t t_start, t_stop;

	#if AS_DEBUG
	UART_Printf(debug, "Demo1 is starting\r\n");
	#endif

  char* graph_data = (char*)(build_test_graph_json);
  char* params_data = (char*)(build_test_params_bin);
  uint64_t params_size = build_test_params_bin_len;
  TVMInit(params_data, params_size, graph_data);

	for (;;) {
		__asm__("wfi");
		InvokeDeferredProcs();
	}

failed:
	for(;;) {
		count++;
		if ((count % 1000000) == 0) {
			#if AS_DEBUG
			UART_Print(debug, "failed\n");
			#endif
		}
	}
}