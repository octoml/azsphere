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

#include "bundle.h"
#include "network.h"
#include "utils.h"

// Convolution
#define in_dim0     1
#define in_dim1     3
#define in_dim2     32
#define in_dim3     32

#define out_dim0    1
#define out_dim1    10

static ExitCode exitCode = ExitCode_Success;
#define interface     "eth0"
#define serverPort    11000
#define serverIP      "192.168.0.10"
#define data_file     "build/cifar_data.bin"
#define output_file   "build/cifar_output.bin"
#define id_file       "build/id.bin"
#define param_file    "build/cifar_params.bin"
#define graph_file    "build/cifar_graph.bin"
static uint16_t id;

int main(int argc, char **argv) {
  #if AS_DEBUG
  Log_Debug("Starting TVM Conv2d Test...\n");
  #endif  /* AS_DEBUG */

  struct timeval t0, t1, t2, t3, t4, t5;
  gettimeofday(&t0, 0);

  int fd = GPIO_OpenAsOutput(SAMPLE_LED, GPIO_OutputMode_PushPull, GPIO_Value_High);
  if (fd < 0) {
    #if AS_DEBUG
    Log_Debug(
        "Error opening GPIO: %s (%d). Check that app_manifest.json includes the GPIO used.\n",
        strerror(errno), errno);
    #endif  /* AS_DEBUG */
    return ExitCode_Main_Led;
  }
  GPIO_SetValue(fd, GPIO_Value_High);


  exitCode = NetworkEnable(interface);
  exitCode = ConfigureNetworkInterfaceWithStaticIp(interface,
                                                 "192.168.0.20",
                                                 "255.255.255.0",
                                                 "192.168.0.1");

  int socket = OpenIpV4Socket(serverIP, serverPort, SOCK_STREAM, &exitCode);

  // Read id
  ReadID(id_file, &id);

  char msg [20];
  int len;
  len = message(id, Message_START, msg);
  send(socket , msg , (size_t)len, 0);

  // Read params
  char* params_data;
  uint64_t params_size = Read_File_Char(param_file, &params_data);
  
  // Read graph
  char* graph_data;
  Read_File_Char(graph_file, &graph_data);

  gettimeofday(&t0, 0);
  auto *handle = tvm_runtime_create(graph_data, params_data, params_size);
  gettimeofday(&t1, 0);

  // Read data
  float* input_storage;
  Read_File_Float(data_file, &input_storage);

  DLTensor input;
  input.data = input_storage;
  DLContext ctx = {kDLCPU, 0};
  input.ctx = ctx;
  input.ndim = 4;
  DLDataType dtype = {kDLFloat, 32, 1};
  input.dtype = dtype;
  int64_t shape [4] = {in_dim0, in_dim1, in_dim2, in_dim3};
  input.shape = shape;
  input.strides = NULL;
  input.byte_offset = 0;

  tvm_runtime_set_input(handle, "conv2d_1_input", &input);
  gettimeofday(&t2, 0);

  tvm_runtime_run(handle);
  gettimeofday(&t3, 0);

  free(params_data);
  free(input_storage);
  free(graph_data);
  // free(&input);

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
  gettimeofday(&t4, 0);

  tvm_runtime_destroy(handle);
  gettimeofday(&t5, 0);

// Read expected output
  float* exp_out;
  Read_File_Float(output_file, &exp_out);

  bool result = true;
  int output_size = out_dim0 * out_dim1;
  for (int i = 0; i < output_size; ++i) {
    if (fabs(output_storage[i] - exp_out[i]) >= 1e-3f) {
      result = false;
      #if AS_DEBUG
      Log_Debug("got %f, expected %f\n", output_storage[i], exp_out[i]);
      #endif  /* AS_DEBUG */
      break;
    }
  }

  len = message(id, Message_RESULT, msg);
  msg[len] = ',';
  if (result) {
    msg[len+1] = '1';
  }
  else{
    msg[len+1] = '0';
  }

  msg[len+2] = '\n';
  len += 3;
  send(socket , msg , (size_t)len, 0);

  #if AS_DEBUG
  Log_Debug("timing: %.2f ms (create), %.2f ms (set_input), %.2f ms (run), "
         "%.2f ms (get_output), %.2f ms (destroy)\n",
         (float)(t1.tv_sec-t0.tv_sec)*1000 + (float)(t1.tv_usec-t0.tv_usec)/1000.f,
         (float)(t2.tv_sec-t1.tv_sec)*1000 + (float)(t2.tv_usec-t1.tv_usec)/1000.f,
         (float)(t3.tv_sec-t2.tv_sec)*1000 + (float)(t3.tv_usec-t2.tv_usec)/1000.f,
         (float)(t4.tv_sec-t3.tv_sec)*1000 + (float)(t4.tv_usec-t3.tv_usec)/1000.f,
         (float)(t5.tv_sec-t4.tv_sec)*1000 + (float)(t5.tv_usec-t4.tv_usec)/1000.f);
  #endif  /* AS_DEBUG */

  float duration = (float)(t3.tv_sec-t2.tv_sec)*1000 + (float)(t3.tv_usec-t2.tv_usec)/1000.f;
  len = message(id, Message_TIME, msg);
  msg[len] = ',';
  len += 1;

  char time_array[10];
  int ret = snprintf(time_array, sizeof time_array, "%3.5f", duration);
  if (ret < 0){
    goto endApp;
  }

  for (int ii=0; ii<10; ii++) {
    msg[len + ii] = time_array[ii];
  }
  len += 10;
  
  msg[len]  = '\n';
  len += 1;
  send(socket , msg , (size_t)len, 0);
  shutdown(socket, 2);

  const struct timespec sleepTime = {.tv_sec = 1, .tv_nsec = 0};
  while (true) {
      GPIO_SetValue(fd, GPIO_Value_Low);
      nanosleep(&sleepTime, NULL);
      GPIO_SetValue(fd, GPIO_Value_High);
      nanosleep(&sleepTime, NULL);
  }

endApp:
  return 0;
}