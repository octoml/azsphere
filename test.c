#include <assert.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include <applibs/log.h>
#include <applibs/gpio.h>
#include <applibs/storage.h>

#include "include/config.h"
#include "include/bundle.h"
#include "include/utils.h"
#include "include/exitcode.h"

// #include <tvm/runtime/c_runtime_api.h>

#define in_dim0     10
#define in_dim1     5

#define out_dim0    10
#define out_dim1    5

#define data_file     "build/test_data.bin"
#define output_file   "build/test_output.bin"
#define param_file    "build/test_params.bin"
#define graph_file    "build/test_graph.json"

int main(int argc, char **argv) {
  fprintf(stdout, "TVM a+b test...\n");
  
  int fd = GPIO_OpenAsOutput(LED1[0], GPIO_OutputMode_PushPull, GPIO_Value_High);
  if (fd < 0) {
    #if AS_DEBUG
    fprintf(stdout,
      "ERROR: opening GPIO: %s (%d). Check that app_manifest.json includes the GPIO used.\n",
      strerror(errno), errno);
    #endif
    return ExitCode_Main_Led;
  }
  int fid;
  struct timeval t0, t1, t2, t3, t4, t5;

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

  //graph and params not required anymore
  free(graph_data);
  free(params_data);

  // Read data
  float* input_storage;
  Read_File_Float(data_file, &input_storage);
  
  DLTensor input;
  input.data = input_storage;
  DLContext ctx = {kDLCPU, 0};
  input.ctx = ctx;
  input.ndim = 2;
  DLDataType dtype = {kDLFloat, 32, 1};
  input.dtype = dtype;
  int64_t shape [2] = {in_dim0, in_dim1};
  input.shape = shape;
  input.strides = NULL;
  input.byte_offset = 0;

  tvm_runtime_set_input(handle, "x", &input);
  gettimeofday(&t2, 0);

  tvm_runtime_run(handle);
  gettimeofday(&t3, 0);

  float output_storage[out_dim0 * out_dim1];
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

  // Read output
  float* exp_out;
  Read_File_Float(output_file, &exp_out);

  bool result = true;
  for (int i = 0; i < (out_dim0 * out_dim1); ++i) {
    assert(fabs(output_storage[i] - exp_out[i]) < 1e-5f);
    if (fabs(output_storage[i] - exp_out[i]) >= 1e-5f) {
      result = false;
      break;
    }
  }

  #if AS_DEBUG
  fprintf(stdout, "timing: %.2f ms (create), %.2f ms (set_input), %.2f ms (run), "
    "%.2f ms (get_output), %.2f ms (destroy)\n",
    (t1.tv_sec-t0.tv_sec)*1000 + (t1.tv_usec-t0.tv_usec)/1000.f,
    (t2.tv_sec-t1.tv_sec)*1000 + (t2.tv_usec-t1.tv_usec)/1000.f,
    (t3.tv_sec-t2.tv_sec)*1000 + (t3.tv_usec-t2.tv_usec)/1000.f,
    (t4.tv_sec-t3.tv_sec)*1000 + (t4.tv_usec-t3.tv_usec)/1000.f,
    (t5.tv_sec-t4.tv_sec)*1000 + (t5.tv_usec-t4.tv_usec)/1000.f);
  #endif

  if (result) {
    GPIO_SetValue(fd, GPIO_Value_Low);
  }
  for(;;) {

  }

failed:
  GPIO_SetValue(fd, GPIO_Value_High);
  return 0;
}
