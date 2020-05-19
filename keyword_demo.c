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

#define NUM_EXP     1

static ExitCode exitCode = ExitCode_Success;
#define interface     "eth0"
#define serverPort    11000
#define serverIP      "192.168.0.10"
#define id_file       "build/id.bin"
#define param_file    "build/keyword_params.bin"
#define graph_file    "build/keyword_graph.bin"
static uint16_t id = 0;
static char * labels [12] = {"_silence_", "_unknown_", "yes", "no", "up", "down",
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
  case 2: //yes
    leds = (uint8_t[12]){0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0};
    break;
  case 3: //no
    leds = (uint8_t[12]){1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
    break;
  case 4: //up
    leds = (uint8_t[12]){0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1};
    break;
  case 5: //down
    leds = (uint8_t[12]){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    break;
  case 6: //left
    leds = (uint8_t[12]){1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    break;
  case 7: //right
    leds = (uint8_t[12]){0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0};
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

int main(int argc, char **argv) {
  #if AS_NETWORKING
  exitCode = NetworkEnable(interface);
  exitCode = ConfigureNetworkInterfaceWithStaticIp(interface,
                                                 "192.168.0.20",
                                                 "255.255.255.0",
                                                 "192.168.0.1");

  int socket = OpenIpV4Socket(serverIP, serverPort, SOCK_STREAM, &exitCode);
  #endif
  Debug_Init(socket);
  fprintf(stdout, "Keyword Demo starting...\n");
  GPIO_Init();

  struct timeval t0, t1;
  float duration;
  const struct timespec sleepTime = {.tv_sec = 1, .tv_nsec = 0};

  #if AS_NETWORKING
  char msg [20];
  int len;
  len = message(id, Message_START, msg);
  send(socket , msg , (size_t)len, 0);
  #endif

  //TVM
  char* params_data;
  uint64_t params_size = Read_File_Char(param_file, &params_data);
  char* graph_data;
  Read_File_Char(graph_file, &graph_data);
  gettimeofday(&t0, 0);
  auto *handle = tvm_runtime_create(graph_data, params_data, params_size);
  gettimeofday(&t1, 0);
  free(graph_data);
  free(params_data);

  int input_size = in_dim0 * in_dim1 * in_dim2 * sizeof(float);
  char * buffer = (char *)malloc(input_size);
  float* input_storage;
  int valread;

  while(1) {
    LED_Set(20);
    memset(buffer, 0, input_size);
    len = message(id, Message_READY, msg);
    send(socket , msg , (size_t)len, 0);
    valread = read(socket, &buffer[0], (size_t)input_size);
    
    if (valread < input_size) {
      // #if AS_DEBUG
      fprintf(stdout, "input read: %d\n", valread);
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

      tvm_runtime_set_input(handle, "Mfcc", &input);
      gettimeofday(&t0, 0);
      tvm_runtime_run(handle);
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

      tvm_runtime_get_output(handle, 0, &output);
      //TODO: add this to final close
      // tvm_runtime_destroy(handle);

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
    }
  }

endApp:
  return 0;
}