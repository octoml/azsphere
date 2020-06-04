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

#include <tvm/runtime/c_runtime_api.h>
#include "include/config.h"
#include "include/intercore.h"
#include "include/exitcode.h"
#include "include/tvmruntime.h"
#include "include/utils.h"

#if AS_NETWORKING
#include "include/network.h"
#endif

#define rtAppComponentId "18b8807b-541b-4953-8c9f-9135eedcc376"
#define id_file       "build/id.bin"
#define param_file    "build/keyword_params.bin"
#define graph_file    "build/keyword_graph.bin"
#define data_file     "build/keyword_data.bin"
#define output_file   "build/keyword_output.bin"

static int sockFd = -1;
static EventLoop *eventLoop = NULL;
static EventLoopTimer *sendTimer = NULL;
static EventRegistration *socketEventReg = NULL;
// static volatile sig_atomic_t exitCode = ExitCode_Success;
static ExitCode exitCode = ExitCode_Success;
int8_t InterCoreRXBuff[InterCoreRXBuffSize];
volatile bool InterCoreRXFlag;
int* tvm_handle;
static char * labels [12] = {"silence", "unknown", "yes", "no", "up", "down",
                          "left", "right", "on", "off", "stop", "go"};

static int led1[3];
static int led2[3];
static int led3[3];
static int led4[3];

// static void TerminationHandler(int signalNumber);
// static void SendTimerEventHandler(EventLoopTimer *timer);
// static void CloseHandlers(void);
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
      Log_Debug(
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
  case 0: //silence
    leds = (uint8_t[12]){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    break;
  case 1: //unknown
    leds = (uint8_t[12]){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    break;
  case 2: //yes
    leds = (uint8_t[12]){0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0};
    break;
  case 3: //no
    leds = (uint8_t[12]){1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
    break;
  case 4: //up
    leds = (uint8_t[12]){0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0};
    break;
  case 5: //down
    leds = (uint8_t[12]){0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0};
    break;
  case 6: //left
    leds = (uint8_t[12]){1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0};
    break;
  case 7: //right
    leds = (uint8_t[12]){0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    break;
  case 8: //on
    leds = (uint8_t[12]){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    break;
  case 9: //off
    leds = (uint8_t[12]){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    break;
  case 10: //stop
    leds = (uint8_t[12]){1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
    break;
  case 11: //go
    leds = (uint8_t[12]){0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0};
    break;
  default:
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
  ExitCode tmp = ExitCode_Success;
  eventLoop = EventLoop_Create();
  if (eventLoop == NULL) {
    fprintf(stdout, "Could not create event loop.\n");
    return ExitCode_Init_EventLoop;
  }

  tmp = InterCoreInit(eventLoop, socketEventReg, sockFd, rtAppComponentId);
  InterCoreRXFlag = false;

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
    // Continue if interrupted by signal, e.g. due to breakpoint being set.

    if (InterCoreRXFlag) {
      InterCoreRXFlag = false;
      Log_Debug("RX: ");
      for(int i=0; i<490; i++){
        Log_Debug("%d, ", InterCoreRXBuff[i]);
      }
      float* input_storage = (float *)malloc(in_dim0*in_dim1*in_dim2*sizeof(float));
      float* tvmOutput = (float *)malloc(out_dim0 * out_dim1 * sizeof(float));
      for (int i=0; i<InterCoreRXBuffSize; i++) {
        input_storage[i] = (float)InterCoreRXBuff[i];
      }
      // Read_File_Float(data_file, &input_storage);
      TVMCallback(tvm_handle, input_storage, tvmOutput);

      // Read expected output
      float* exp_out;
      Read_File_Float(output_file, &exp_out);

      bool result = true;
      int output_size = out_dim0 * out_dim1;
      for (int i = 0; i < output_size; ++i) {
        if (fabs(tvmOutput[i] - exp_out[i]) >= 1e-3f) {
          result = false;
          #if AS_DEBUG
          fprintf(stdout, "got %f, expected %f\n", tvmOutput[i], exp_out[i]);
          #endif  /* AS_DEBUG */
          break;
        }
      }
      int index = TVMMaxIndex(tvmOutput);
      Log_Debug(stdout, "label: %s\n", labels[index]);
      // LED_Set(index);

      // Log_Debug("Intercore inside\n");
    }

    if (result == EventLoop_Run_Failed && errno != EINTR) {
        exitCode = ExitCode_Main_EventLoopFail;
    }
  }

  // CloseHandlers();
  // while(1){
  //   ;;
  // }
  return exitCode;
}
