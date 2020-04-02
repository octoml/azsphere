/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/runtime/c_runtime_api.h>

#include <assert.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <applibs/log.h>
#include <applibs/gpio.h>
//Header files for read-only storage
#include <unistd.h>
#include <applibs/storage.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
// This #include imports the sample_hardware abstraction from that hardware definition.
#include <hw/sample_hardware.h>

#include "bundle.h"
#include "build/conv2d_graph.json.c"
#include "build/conv2d_params.bin.c"

/// <summary>
/// Exit codes for this application. These are used for the
/// application exit code.  They they must all be between zero and 255,
/// where zero is reserved for successful termination.
/// </summary>
typedef enum {
    ExitCode_Success = 0,

    ExitCode_Main_Led = 1
} ExitCode;

// Convolution
#define H       14
#define W       14
#define in_ch   10
#define out_ch  20
#define batch   1

int main(int argc, char **argv) {
  Log_Debug("Starting TVM Conv2d Test...\n");
  assert(argc == 3 && "Usage: conv2d <conv2d_data.bin, conv2d_output.bin>");
  
  int fd = GPIO_OpenAsOutput(SAMPLE_LED, GPIO_OutputMode_PushPull, GPIO_Value_High);
  if (fd < 0) {
    Log_Debug(
        "Error opening GPIO: %s (%d). Check that app_manifest.json includes the GPIO used.\n",
        strerror(errno), errno);
    return ExitCode_Main_Led;
  }
  
  char * json_data = (char *)(build_conv2d_graph_json);
  char * params_data = (char *)(build_conv2d_params_bin);
  uint64_t params_size = build_conv2d_params_bin_len;
  int fid;
  off_t fs;

  struct timeval t0, t1, t2, t3, t4, t5;
  gettimeofday(&t0, 0);

  auto *handle = tvm_runtime_create(json_data, params_data, params_size);
  gettimeofday(&t1, 0);

  // Read data
  float input_storage[batch * in_ch * H * W];
  fid = Storage_OpenFileInImagePackage(argv[1]);
  if (fid == -1) {
    Log_Debug("Error: Openning %s failed!", argv[1]);
    goto failed;
  }
  fs = lseek(fid, 0, SEEK_END);
  if (fs == -1) {
    Log_Debug("Error: File %s size!", argv[1]);
    goto failed;
  }
  lseek(fid, 0, SEEK_SET);
  Log_Debug("%s size: %d", argv[1], (int)fs);
  read(fid, &input_storage, fs);
  close(fid);

  DLTensor input;
  input.data = input_storage;
  DLContext ctx = {kDLCPU, 0};
  input.ctx = ctx;
  input.ndim = 4;
  DLDataType dtype = {kDLFloat, 32, 1};
  input.dtype = dtype;
  int64_t shape [4] = {batch, in_ch, H, W};
  input.shape = &shape;
  input.strides = NULL;
  input.byte_offset = 0;

  tvm_runtime_set_input(handle, "A", &input);
  gettimeofday(&t2, 0);

  tvm_runtime_run(handle);
  gettimeofday(&t3, 0);

  float output_storage[batch * out_ch * H * W];
  DLTensor output;
  output.data = output_storage;
  DLContext out_ctx = {kDLCPU, 0};
  output.ctx = out_ctx;
  output.ndim = 4;
  DLDataType out_dtype = {kDLFloat, 32, 1};
  output.dtype = out_dtype;
  int64_t out_shape [4] = {batch, out_ch, H, W};
  output.shape = &out_shape;
  output.strides = NULL;
  output.byte_offset = 0;
  
  tvm_runtime_get_output(handle, 0, &output);
  gettimeofday(&t4, 0);

  tvm_runtime_destroy(handle);
  gettimeofday(&t5, 0);

// Read expected output
  float exp_out[batch * out_ch * H * W];
  fid = Storage_OpenFileInImagePackage(argv[2]);
  if (fid == -1) {
    Log_Debug("Error: Openning %s failed!", argv[2]);
    goto failed;
  }
  fs = lseek(fid, 0, SEEK_END);
  if (fs == -1) {
    Log_Debug("Error: File %s size!", argv[2]);
    goto failed;
  }
  lseek(fid, 0, SEEK_SET);
  Log_Debug("%s size: %d", argv[2], (int)fs);
  read(fid, &exp_out, fs);
  close(fid);

  for (auto i = 0; i < batch*out_ch*H*W; ++i) {
    assert(fabs(output_storage[i] - exp_out[i]) < 1e-5f);
    if (fabs(output_storage[i] - exp_out[i]) >= 1e-5f) {
      Log_Debug("got %f, expected %f\n", output_storage[i], exp_out[i]);
    }
  }

  Log_Debug("timing: %.2f ms (create), %.2f ms (set_input), %.2f ms (run), "
         "%.2f ms (get_output), %.2f ms (destroy)\n",
         (t1.tv_sec-t0.tv_sec)*1000000 + (t1.tv_usec-t0.tv_usec)/1000.f,
         (t2.tv_sec-t1.tv_sec)*1000000 + (t2.tv_usec-t1.tv_usec)/1000.f,
         (t3.tv_sec-t2.tv_sec)*1000000 + (t3.tv_usec-t2.tv_usec)/1000.f,
         (t4.tv_sec-t3.tv_sec)*1000000 + (t4.tv_usec-t3.tv_usec)/1000.f,
         (t5.tv_sec-t4.tv_sec)*1000000 + (t5.tv_usec-t4.tv_usec)/1000.f);
  
  const struct timespec sleepTime = {.tv_sec = 1, .tv_nsec = 0};
  while (true) {
      GPIO_SetValue(fd, GPIO_Value_Low);
      nanosleep(&sleepTime, NULL);
      GPIO_SetValue(fd, GPIO_Value_High);
      nanosleep(&sleepTime, NULL);
  }

failed:
  return 0;
}
