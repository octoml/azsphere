#include <sys/time.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <applibs/log.h>
#include <applibs/gpio.h>
#include <applibs/networking.h>
#include <applibs/storage.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <hw/sample_hardware.h>
#include <tvm/runtime/c_runtime_api.h>
#include <float.h>
#include <signal.h>

#include "config.c"
#include "bundle.h"
#include "utils.h"
#include "exitcode.h"

#if AS_NETWORKING
#include "network.h"
#endif

// Convolution
#define in_dim0     1
#define in_dim1     49
#define in_dim2     10

#define out_dim0    1
#define out_dim1    12

static ExitCode exitCode = ExitCode_Success;
#define interface     "eth0"
#define serverPort    11000
#define serverIP      "192.168.0.10"
#define staticIP      "192.168.0.20"
#define subnet        "255.255.255.0"
#define gateway       "192.168.0.1"
#define id_file       "build/id.bin"
#define param_file    "build/keyword_params.bin"
#define graph_file    "build/keyword_graph.bin"
static int server_socket;
static uint16_t id = 0;
static char * labels [12] = {"silence", "unknown", "yes", "no", "up", "down",
                          "left", "right", "on", "off", "stop", "go"};
static int led1[3];
static int led2[3];
static int led3[3];
static int led4[3];

static int GPIO_Init() {
  for(int ii=0; ii<3; ii++) {
    led1[ii] = GPIO_OpenAsOutput(LED1[ii], GPIO_OutputMode_PushPull, GPIO_Value_High);
    led2[ii] = GPIO_OpenAsOutput(LED2[ii], GPIO_OutputMode_PushPull, GPIO_Value_High);
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

static int LED_Set(int label) {
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
    GPIO_SetValue(led2[ii], leds[ii+3]^1);
    GPIO_SetValue(led3[ii], leds[ii+6]^1);
    GPIO_SetValue(led4[ii], leds[ii+9]^1);
  }
}

static void TerminationHandler(int signalNumber)
{
  exitCode = ExitCode_TermHandler_SigTerm;
}

static ExitCode Initialize() {
  ExitCode result = ExitCode_Success;
  #if AS_NETWORKING
  result = NetworkEnable(interface);
  result = ConfigureNetworkInterfaceWithStaticIp(interface, staticIP,
                                                 subnet, gateway);
  server_socket = OpenIpV4Socket(serverIP, serverPort, SOCK_STREAM, &exitCode);
  if (server_socket < 0) {
    result = ExitCode_OpenIpV4_Socket;
  }

  #endif
  #if AS_ETHERNET_DEBUG
  Debug_Init(server_socket);
  #endif

  struct sigaction action;
  memset(&action, 0, sizeof(struct sigaction));
  action.sa_handler = TerminationHandler;
  sigaction(SIGTERM, &action, NULL);

  GPIO_Init();
  LED_Set(20);
  return result;
}

static void ShutDownAndCleanup(void)
{
    CloseFdAndPrintError(server_socket, "Socket");
}

#include "wav_data.h"
#include "mfcc.h"

int main(int argc, char **argv) {
  Initialize();
  fprintf(stdout, "Keyword Demo starting...\n");
  
  struct timeval t0, t1;
  float duration;
  const struct timespec sleepTime = {.tv_sec = 1, .tv_nsec = 0};
  int * tvm_handle;

  #if AS_NETWORKING
  char msg [20];
  int len;
  len = message(id, Message_START, msg);
  send(server_socket , msg , (size_t)len, 0);
  #endif

  //TVM
  char* params_data;
  uint64_t params_size = Read_File_Char(param_file, &params_data);
  fprintf(stdout, "param read\n");

  char* graph_data;
  Read_File_Char(graph_file, &graph_data);

  fprintf(stdout, "graph read\n");

  gettimeofday(&t0, 0);
  tvm_handle = tvm_runtime_create(graph_data, params_data, params_size);
  gettimeofday(&t1, 0);
  free(graph_data);
  free(params_data);

  fprintf(stdout, "handle created\n");


  int input_size = in_dim0 * in_dim1 * in_dim2 * sizeof(float);
  char * buffer = (char *)malloc(input_size);
  float* input_storage;
  int valread;
  uint16_t recInd;
  size_t readBlock = 1;
  int status = -1;

  while(1) {
    memset(buffer, 0, input_size);
    len = message(id, Message_READY, msg);
    status = send(server_socket , msg , (size_t)len, 0);

    recInd = 0;
    valread = 0;
    while((valread >= 0) && (recInd < input_size)) {
      valread = read(server_socket, &buffer[recInd], readBlock);
      if (valread >= 0) {
        recInd = recInd + valread;
      }
    }

    // valread = read(server_socket, &buffer[0], (size_t)input_size);
    
    if (valread < 0) {
      // #if AS_DEBUG
      fprintf(stdout, "Error: reading input data!\n");
      // #endif
    }
    else {
      input_storage = (float *)buffer;
      DLTensor input;
      input.data = input_storage;
      DLContext ctx = {kDLCPU, 0};
      input.ctx = ctx;
      input.ndim = 3;
      DLDataType dtype = {kDLFloat, 32, 1};
      input.dtype = dtype;
      int64_t shape [3] = {in_dim0, in_dim1, in_dim2};
      input.shape = shape;
      input.strides = NULL;
      input.byte_offset = 0;

      tvm_runtime_set_input(tvm_handle, "Mfcc", &input);
      gettimeofday(&t0, 0);
      tvm_runtime_run(tvm_handle);
      gettimeofday(&t1, 0);
      duration = (float)(t1.tv_sec-t0.tv_sec)*1000 + (float)(t1.tv_usec-t0.tv_usec)/1000.f;

      float* output_storage = malloc(out_dim0 * out_dim1 * sizeof(float));
      DLTensor output;
      output.data = output_storage;
      DLContext out_ctx = {kDLCPU, 0};
      output.ctx = out_ctx;
      output.ndim = 2;
      DLDataType out_dtype = {kDLFloat, 32, 1};
      output.dtype = out_dtype;
      int64_t out_shape [2] = {out_dim0, out_dim1};
      output.shape = out_shape;
      output.strides = NULL;
      output.byte_offset = 0;

      tvm_runtime_get_output(tvm_handle, 0, &output);

      float max_iter = -FLT_MAX;
      int32_t max_index = -1;
      for (int i = 0; i < out_dim1; ++i) {
        if (output_storage[i] > max_iter) {
          max_iter = output_storage[i];
          max_index = i;
        }
      }
      fprintf(stdout, "label: %s, runtime: %3.2f\n", labels[max_index], duration);
      LED_Set(max_index);
      // nanosleep(&sleepTime, NULL);
    }

    if (status < 0) {
      server_socket = OpenIpV4Socket(serverIP, serverPort, SOCK_STREAM, &exitCode);
    }
  }

  tvm_runtime_destroy(tvm_handle);

  ShutDownAndCleanup();
  return exitCode;
}