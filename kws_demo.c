#include <signal.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <stdbool.h>
#include <errno.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <sys/socket.h>

#include <applibs/log.h>
#include <applibs/gpio.h>
#include <applibs/application.h>

#include "include/eventloop_timer_utilities.h"
#include "include/config.h"
#include "include/intercore.h"
#include "include/exitcode.h"
#include "include/tvmruntime.h"
#include "include/utils.h"

#if AS_NETWORKING
#include "include/network.h"
#endif

#define rtAppComponentId "18b8807b-541b-4953-8c9f-9135eedcc376"
#define param_file    "build/keyword_params.bin"
#define graph_file    "build/keyword_graph.bin"

static int sockFd = -1;
static EventLoop *eventLoop = NULL;
static EventLoopTimer *sendTimer = NULL;
static EventRegistration *socketEventReg = NULL;
static ExitCode exitCode = ExitCode_Success;
float InterCoreRXBuff[InterCoreRXBuffSize];
volatile uint32_t InterCoreRXIndex;
volatile int intercore_counter;
int* tvm_handle;
static char * labels [12] = {"silence", "unknown", "yes", "no", "up", "down",
                          "left", "right", "on", "off", "stop", "go"};

static int led1[3];
static int led2[3];
static int led3[3];
static int led4[3];

static ExitCode GPIO_Init();
static int LED_Set(uint8_t label);

static ExitCode GPIO_Init() {
  for(int ii=0; ii<3; ii++) {
    led1[ii] = GPIO_OpenAsOutput(LED1[ii], GPIO_OutputMode_PushPull, GPIO_Value_High);
    // led2[ii] = GPIO_OpenAsOutput(LED2[ii], GPIO_OutputMode_PushPull, GPIO_Value_High);
    led3[ii] = GPIO_OpenAsOutput(LED3[ii], GPIO_OutputMode_PushPull, GPIO_Value_High);
    led4[ii] = GPIO_OpenAsOutput(LED4[ii], GPIO_OutputMode_PushPull, GPIO_Value_High);
  }

  for(int ii=0; ii<4; ii++){
    if ((led1[ii] < 0) || (led2[ii] < 0) || (led3[ii] < 0) || (led4[ii] < 0)){
      #if AS_DEBUG
      fprintf(stdout,
      "Error opening GPIO: %s (%d). Check that app_manifest.json includes the GPIO used.\n",
      strerror(errno), errno);
      #endif
      return ExitCode_Main_Led;
    }
  }
  return ExitCode_Success;
}

static int LED_Set(uint8_t label) {
  uint8_t *leds;
  switch (label)
  {
  case 0: //silence, off, off, off, blue
    leds = (uint8_t[12]){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    break;
  case 1: //unknown => off, off, off, green
    leds = (uint8_t[12]){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    break;
  case 2: //yes => off, off, green, green
    leds = (uint8_t[12]){0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0};
    break;
  case 3: //no => off, off, red, red
    leds = (uint8_t[12]){0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0};
    break;
  case 4: //up => off, off, green, off
    leds = (uint8_t[12]){0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0};
    break;
  case 5: //down => off, off, blue, off
    leds = (uint8_t[12]){0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0};
    break;
  case 6: //left => rgb, off, off, off
    leds = (uint8_t[12]){1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    break;
  case 7: //right => off, off, off, rgb
    leds = (uint8_t[12]){0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1};
    break;
  case 8: //on => rgb, off, rgb, rgb
    leds = (uint8_t[12]){1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    break;
  case 9: //off => red, off, off, off
    leds = (uint8_t[12]){1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    break;
  case 10: //stop => red, off, red, red
    leds = (uint8_t[12]){1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0};
    break;
  case 11: //go => green, off, green, green
    leds = (uint8_t[12]){0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0};
    break;
  default: //=> off, off, off, off
    leds = (uint8_t[12]){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    break;
  }
  for(uint8_t ii=0; ii<3; ii++) {
    GPIO_SetValue(led1[ii], leds[ii]^1);
    // GPIO_SetValue(led2[ii], leds[ii+3]^1);
    GPIO_SetValue(led3[ii], leds[ii+6]^1);
    GPIO_SetValue(led4[ii], leds[ii+9]^1);
  }
}

// static void CloseHandlers(void)
// {
//   DisposeEventLoopTimer(sendTimer);
//   EventLoop_UnregisterIo(eventLoop, socketEventReg);
//   EventLoop_Close(eventLoop);

//   Log_Debug("Closing file descriptors.\n");
//   CloseFdAndPrintError(sockFd, "Socket");
// }

static ExitCode App_Init() {
  GPIO_Init();
  LED_Set(20);
  ExitCode tmp = ExitCode_Success;
  eventLoop = EventLoop_Create();
  if (eventLoop == NULL) {
    fprintf(stdout, "Could not create event loop.\n");
    return ExitCode_Init_EventLoop;
  }

  tmp = InterCoreInit(eventLoop, socketEventReg, sockFd, rtAppComponentId);
  InterCoreRXIndex = 0;

  tvm_handle = TVMInit(param_file, graph_file);
  if (tvm_handle < 0) {
    return ExitCode_TVM_Init;
  }
  return tmp;
}

int main(void)
{
  fprintf(stdout, "Demo1 starting...\n");
  exitCode = App_Init();

  while (exitCode == ExitCode_Success) {
    EventLoop_Run_Result result = EventLoop_Run(eventLoop, -1, true);
    if (InterCoreRXIndex >= (InterCoreRXBuffSize*sizeof(float))) {
      InterCoreRXIndex = 0;
      intercore_counter = 0;
      #if AS_DEBUG
      fprintf(stdout, "intercore_counter: %d\n", intercore_counter);
      fprintf(stdout, "RX: ");
      for(int i=0; i<InterCoreRXBuffSize; i++){
        fprintf(stdout, "%d, ", InterCoreRXBuff[i]);
      }
      #endif

      float* tvmOutput = (float *)malloc(out_dim0 * out_dim1 * sizeof(float));
      TVMCallback(tvm_handle, &InterCoreRXBuff, tvmOutput);

      int index = TVMMaxIndex(tvmOutput);
      #if AS_DEBUG
      fprintf(stdout, "label: %s\n", labels[index]);
      #endif
      LED_Set(index);
    }

    if (result == EventLoop_Run_Failed && errno != EINTR) {
        exitCode = ExitCode_Main_EventLoopFail;
    }
  }
  
  return exitCode;
}
