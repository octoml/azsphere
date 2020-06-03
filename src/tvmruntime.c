#include <stdio.h>
#include <float.h>
#include <stdlib.h>

#include "include/tvmruntime.h"
#include "include/utils.h"
#include "include/exitcode.h"

int* TVMInit(char* paramFile, char* graphFile) {
  int * handle;

  char* params_data;
  int status = Read_File_Char(paramFile, &params_data);
  if (status < 0) {
    return -1;
  }
  uint64_t params_size = (uint64_t)status;

  #ifdef AS_DEBUG
  fprintf(stdout, "param read\n");
  #endif

  char* graph_data;
  Read_File_Char(graphFile, &graph_data);

  #ifdef AS_DEBUG
  fprintf(stdout, "graph read\n");
  #endif

  handle = tvm_runtime_create(graph_data, params_data, params_size);
  free(graph_data);
  free(params_data);
  return handle;
}

ExitCode TVMCallback(int* handle, void* inputData, float* result) {
  int input_size = in_dim0 * in_dim1 * in_dim2 * sizeof(float);
  //TODO: this is an issue
  float* input_storage = (float *)inputData;
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
  tvm_runtime_run(handle);

  // *result = (float *)malloc(out_dim0 * out_dim1 * sizeof(float));
  DLTensor output_tensor;
  output_tensor.data = result;
  DLContext out_ctx = {kDLCPU, 0};
  output_tensor.ctx = out_ctx;
  output_tensor.ndim = 2;
  DLDataType out_dtype = {kDLFloat, 32, 1};
  output_tensor.dtype = out_dtype;
  int64_t out_shape [2] = {out_dim0, out_dim1};
  output_tensor.shape = out_shape;
  output_tensor.strides = NULL;
  output_tensor.byte_offset = 0;

  tvm_runtime_get_output(handle, 0, &output_tensor);
  return ExitCode_Success;
}

uint8_t TVMMaxIndex(float* tvmOutput) {
  float max_iter = -FLT_MAX;
  int32_t max_index = -1;
  for (uint8_t ii = 0; ii<(out_dim0 * out_dim1); ++ii) {
    if (tvmOutput[ii] > max_iter) {
      max_iter = tvmOutput[ii];
      max_index = ii;
    }
  }
  return max_index;
}