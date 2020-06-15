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

#include "include/kws.h"
#include "include/config.h"
#include "include/intercore.h"
#include "python/build/wav_data.h"

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
	InterCoreInit();

	button_timer = GPT_Open(MT3620_UNIT_GPT1, 32768, GPT_MODE_REPEAT);
	if (!button_timer) {
		#if AS_DEBUG
		UART_Print(debug, "ERROR: Failed to open button timer\r\n");
		#endif
	}

	int32_t error = GPT_StartTimeout(button_timer, buttonPressCheckPeriodMs,
		GPT_UNITS_MILLISEC, HandleButtonTimerIrq);
	if (error != ERROR_NONE) {
		#if AS_DEBUG
		UART_Printf(debug, "ERROR(%" PRId32 "): Failed to start button timer\r\n", error);
		#endif
	}

	exec_timer = GPT_Open(MT3620_UNIT_GPT3, exec_timer_freq, GPT_MODE_REPEAT);
	if (!exec_timer) {
		#if AS_DEBUG
		UART_Print(debug, "ERROR: Failed to open execution timer\r\n");
		#endif
	}
	error = GPT_Start_Freerun(exec_timer);
	if (error != ERROR_NONE) {
		#if AS_DEBUG
		UART_Printf(debug, "ERROR(%" PRId32 "): Failed to start execution timer\r\n", error);
		#endif
	}
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

  //Preprocessing Init
  kws* new_kws = KWS_Init();
  if (!new_kws){
    goto failed;
  }

	for (;;) {
		__asm__("wfi");
		InvokeDeferredProcs();

		if (record) {
			#if AS_DEBUG
			UART_Printf(debug, "Inferring...!\r\n");
			#endif

			t_start = GPT_GetRunningTime(exec_timer, GPT_UNITS_MILLISEC);
			int32_t mfcc_buffer_head = 0;
			int16_t* audio_buffer;
			int audio_buffer_counter = 0;
			while (mfcc_buffer_head < (new_kws->num_mfcc_features*new_kws->recording_win)) {
				audio_buffer = (int16_t *)(WAVE_DATA + (audio_buffer_counter * new_kws->frame_shift));
      	audio_buffer_counter += new_kws->num_frames;

				int8_t mfcc_status = KWS_Extract_Features_Frame(new_kws, audio_buffer, mfcc_buffer_head);
				if (mfcc_status < 0) {
					goto failed;
				}
				mfcc_buffer_head += 30;
			}
			t_stop = GPT_GetRunningTime(exec_timer, GPT_UNITS_MILLISEC);
			UART_Printf(debug, "Execution time: %d ms\r\n", t_stop-t_start);

			#if AS_DEBUG
			UART_Print(debug, "Extract Features Done!\r\n");
			for (int ii=0; ii<(new_kws->mfcc_buffer_size); ii++){
				UART_Printf(debug, "%f\r\n", new_kws->mfcc_buffer[ii]);
			}
			#endif
			InterCoreMessageSend(&HLCompID, new_kws->mfcc_buffer, new_kws->mfcc_buffer_size * sizeof(float));
			record = false;
		}
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