#include <assert.h>
#include <sys/time.h>
#include <math.h>
#include <applibs/gpio.h>

#include "include/config.h"
#include "include/bundle.h"
#include "include/utils.h"
#include "include/exitcode.h"
#include "include/network.h"

#define in_dim0     1
#define in_dim1     3
#define in_dim2     8
#define in_dim3     8

#define out_dim0    1
#define out_dim1    16     
#define out_dim2    8
#define out_dim3    8

static ExitCode exitCode = ExitCode_Success;
#define interface     "eth0"
#define serverPort    11000
#define serverIP      "192.168.0.10"
#define data_file     "build/conv2d_data.bin"
#define output_file   "build/conv2d_output.bin"
#define id_file       "build/id.bin"
#define param_file    "build/conv2d_params.bin"
#define graph_file    "build/conv2d_graph.json"
static uint16_t id;

int main(int argc, char **argv) {
  #if AS_NETWORKING
  exitCode = NetworkEnable(interface);
  exitCode = ConfigureNetworkInterfaceWithStaticIp(interface,
                                                 "192.168.0.20",
                                                 "255.255.255.0",
                                                 "192.168.0.1");

  int socket = OpenIpV4Socket(serverIP, serverPort, SOCK_STREAM, &exitCode);
  #endif
  #if AS_NETWORK_DEBUG
  Debug_Init(socket);
  #endif
  fprintf(stdout, "TVM Conv2d Test with network...\n");

  int fd = GPIO_OpenAsOutput(LED1[1], GPIO_OutputMode_PushPull, GPIO_Value_High);
  if (fd < 0) {
    #if AS_DEBUG
    fprintf(stdout,
      "Error opening GPIO: %s (%d). Check that app_manifest.json includes the GPIO used.\n",
      strerror(errno), errno);
    #endif
    goto failed;
  }

  struct timeval t0, t1, t2, t3, t4, t5;
  gettimeofday(&t0, 0);

  // Read id
  ReadID(id_file, &id);

  #if AS_NETWORKING
  char msg [20];
  int len;
  len = message(id, Message_START, msg);
  send(socket , msg , (size_t)len, 0);
  #endif
  
  // Read params
  char* params_data;
  uint64_t params_size;
  int error = Read_File_Char(param_file, &params_data);
  if (error < 0){
    #if AS_DEBUG
    fprintf(stdout, "ERROR: reading param file.\n");
    #endif
    goto failed;
  }
  params_size = (uint64_t)error;

  // Read graph
  char* graph_data;
  Read_File_Char(graph_file, &graph_data);

  gettimeofday(&t0, 0);
  int *handle = tvm_runtime_create(graph_data, params_data, params_size);
  gettimeofday(&t1, 0);

  free(graph_data);
  free(params_data);

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

  tvm_runtime_set_input(handle, "A", &input);
  gettimeofday(&t2, 0);

  tvm_runtime_run(handle);
  gettimeofday(&t3, 0);

  float output_storage[out_dim0 * out_dim1 * out_dim2 * out_dim3];
  DLTensor output;
  output.data = output_storage;
  DLContext out_ctx = {kDLCPU, 0};
  output.ctx = out_ctx;
  output.ndim = 4;
  DLDataType out_dtype = {kDLFloat, 32, 1};
  output.dtype = out_dtype;
  int64_t out_shape [4] = {out_dim0, out_dim1, out_dim2, out_dim3};
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
  for (int i = 0; i < (out_dim0*out_dim1*out_dim2*out_dim3); ++i) {
    assert(fabs(output_storage[i] - exp_out[i]) < 1e-5f);
    if (fabs(output_storage[i] - exp_out[i]) >= 1e-5f) {
      result = false;
      break;
    }
  }

  #if AS_NETWORKING
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
  #endif

  #if AS_DEBUG
  fprintf(stdout, "timing: %.2f ms (create), %.2f ms (set_input), %.2f ms (run), "
    "%.2f ms (get_output), %.2f ms (destroy)\n",
    (t1.tv_sec-t0.tv_sec)*1000 + (t1.tv_usec-t0.tv_usec)/1000.f,
    (t2.tv_sec-t1.tv_sec)*1000 + (t2.tv_usec-t1.tv_usec)/1000.f,
    (t3.tv_sec-t2.tv_sec)*1000 + (t3.tv_usec-t2.tv_usec)/1000.f,
    (t4.tv_sec-t3.tv_sec)*1000 + (t4.tv_usec-t3.tv_usec)/1000.f,
    (t5.tv_sec-t4.tv_sec)*1000 + (t5.tv_usec-t4.tv_usec)/1000.f);
  #endif

  #if AS_NETWORKING
  float duration = (t3.tv_sec-t2.tv_sec)*1000 + (t3.tv_usec-t2.tv_usec)/1000.f;
  len = message(id, Message_TIME, msg);
  msg[len] = ',';
  len += 1;

  char time_array[10];
  int ret = snprintf(time_array, sizeof time_array, "%3.5f", duration);
  if (ret < 0){
    goto failed;
  }

  for (int ii=0; ii<10; ii++) {
    msg[len + ii] = time_array[ii];
  }
  len += 10;
  
  msg[len]  = '\n';
  len += 1;
  send(socket , msg , (size_t)len, 0);
  shutdown(socket, 2);
  #endif

  if (result) {
    GPIO_SetValue(fd, GPIO_Value_Low);
  }

  for(;;) {}

failed:
  GPIO_SetValue(fd, GPIO_Value_High);
  return 0;
}