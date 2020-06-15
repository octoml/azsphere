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
#include "lib/ADC.h"
#include "lib/logical-dpc.h"

#include "include/kws.h"
#include "include/config.h"
#include "include/intercore.h"

#define FREQUENCY	197600000
#define UART_BAUD	2000000
#define ADC_DATA_SIZE 1
#define ADC_CHANNELS 1
#define ADC_FREQUENCY 21700
#define ADC_REF	2000
#define ADC_PIN_MASK	0x1
#define ADC_OFFSET 2681
#define AUDIO_BUFF_SIZE	1280

UART *debug = NULL;
static GPT *button_timer = NULL;
static GPT *exec_timer = NULL;
static uint32_t exec_timer_freq = 32768;
static AdcContext *handle = NULL;
static const uint32_t buttonBGpio = 13;
static const uint32_t adc_debug_gpio = 60;
static const uint32_t debug_led = 15;
static const ComponentId HLCompID = {.data1 = 0x1689d8b2,
                                    .data2 = 0xc835,
                                    .data3 = 0x2e27,
                                    .data4 = {0x27, 0xad, 0xe8, 0x94, 0xd6, 0xd1, 0x5f, 0xa9}};
static __attribute__((section(".sysram"))) uint32_t adcRawData[ADC_DATA_SIZE];
bool button_pressed;
static bool adc_started = false;
static ADC_Data adcData[ADC_DATA_SIZE];

int16_t *audio_buffer;
static bool adc_gpio_state = false;
static uint32_t audio_buffer_ind =0;
static int32_t mfcc_buffer_head = 0;

static void AdcCallback(int32_t status);
static void HandleButtonTimerIrq(GPT *);
static void HandleButtonTimerIrqDeferred(void);

void App_Init(void) {
	VectorTableInit();
	CPUFreq_Set(FREQUENCY);
	debug = UART_Open(MT3620_UNIT_UART_DEBUG, UART_BAUD, UART_PARITY_NONE, 1, NULL);
	GPIO_ConfigurePinForInput(buttonBGpio);
	GPIO_ConfigurePinForOutput(debug_led);
	GPIO_Write(debug_led, true);
	GPIO_ConfigurePinForOutput(adc_debug_gpio);
	InterCoreInit();

	button_pressed = false;
	adc_started = false;
	audio_buffer_ind = 0;
	mfcc_buffer_head = 0;
	audio_buffer = malloc(AUDIO_BUFF_SIZE * sizeof(int16_t));
	memset(audio_buffer, 0, AUDIO_BUFF_SIZE * sizeof(int16_t));

	button_timer = GPT_Open(MT3620_UNIT_GPT1, 32768, GPT_MODE_REPEAT);
	if (!button_timer) {
		#if AS_DEBUG
		UART_Print(debug, "ERROR: Failed to open button timer\r\n");
		#endif
	}

	int32_t error = GPT_StartTimeout(button_timer, 10, GPT_UNITS_MILLISEC, HandleButtonTimerIrq);
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

	handle = ADC_Open(MT3620_UNIT_ADC0);
	ADC_ReadPeriodicAsync(handle, &AdcCallback, ADC_DATA_SIZE, adcData, 
		adcRawData, ADC_PIN_MASK, ADC_FREQUENCY, ADC_REF);
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
			button_pressed = true;
		}
		prevState = newState;
	}
}

static void AdcCallback(int32_t status)
{
	adc_gpio_state = !adc_gpio_state;
	GPIO_Write(adc_debug_gpio, adc_gpio_state);
	audio_buffer[audio_buffer_ind] = (int16_t)adcData[0].value - ADC_OFFSET;
	audio_buffer_ind++;
	#if AS_ADC_DEBUG
	UART_Printf(debug, "%d\r\n", adcData[0].value);
	#endif
}

_Noreturn void RTCoreMain(void)
{
	//debugging: change b to false for debugging
	volatile bool b = true;
	uint16_t count = 0;
	bool led2_state = false;
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
		if (button_pressed) {
			#if AS_DEBUG
			UART_Printf(debug, "Pressed\r\n");
			#endif
			audio_buffer_ind = 0;
			mfcc_buffer_head = 0;
			adc_started = true;
			button_pressed = false;
			ADC_Start();
		}

		if (adc_started) {
			if (audio_buffer_ind >= (new_kws->num_frames+1)*(new_kws->frame_len/2)) {
				audio_buffer_ind = 0;
				int8_t mfcc_status = KWS_Extract_Features_Frame(new_kws, audio_buffer, mfcc_buffer_head);
				if (mfcc_status < 0) {
					goto failed;
				}
				audio_buffer_ind = new_kws->frame_len/2;
				memmove(audio_buffer, audio_buffer + (new_kws->num_frames * new_kws->frame_len)/2, new_kws->frame_len/2);
				mfcc_buffer_head += 30;
			}

			if (mfcc_buffer_head >= (new_kws->num_mfcc_features*new_kws->recording_win)) {
				ADC_Stop();
				mfcc_buffer_head = 0;
				adc_started = false;
				InterCoreMessageSend(&HLCompID, new_kws->mfcc_buffer, new_kws->mfcc_buffer_size * sizeof(float));
				memset(new_kws->mfcc_buffer, 0, new_kws->mfcc_buffer_size * sizeof(float));
			}
			continue;
		}

		// __asm__("wfi");
		InvokeDeferredProcs();
	}

failed:
	for(;;) {
		count++;
		if ((count % 1000000) == 0) {
			#if AS_DEBUG
			UART_Print(debug, "failed\n");
			#endif
			GPIO_Write(debug_led, led2_state);
			led2_state = !led2_state;
		}
	}
}