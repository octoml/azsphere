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

#define A_dim1      1
#define A_dim2      64
#define out_dim1    1
#define out_dim2    12

static ExitCode exitCode = ExitCode_Success;
static char interface[] = "eth0";
static uint16_t serverPort = 11000;
static char serverIP[] = "192.168.0.10";
static uint16_t id;
static char data_file[] = "build/conv2d_data.bin";
static char output_file[] = "build/conv2d_output.bin";
static char id_file[] = "build/id.bin";
static char param_file[] = "build/conv2d_params.bin";
static char graph_file[] = "build/conv2d_graph.bin";

int main(int argc, char **argv) {
  Log_Debug("Starting TVM cifar_task0...\n");
  
  struct timeval t0, t1, t2, t3, t4, t5;
  gettimeofday(&t0, 0);
  int fid;
  off_t fs;

  int fd = GPIO_OpenAsOutput(SAMPLE_LED, GPIO_OutputMode_PushPull, GPIO_Value_High);
  if (fd < 0) {
    Log_Debug(
        "Error opening GPIO: %s (%d). Check that app_manifest.json includes the GPIO used.\n",
        strerror(errno), errno);
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
  input.ndim = 2;
  DLDataType dtype = {kDLFloat, 32, 1};
  input.dtype = dtype;
  int64_t shape [2] = {A_dim1, A_dim2};
  input.shape = shape;
  input.strides = NULL;
  input.byte_offset = 0;

  tvm_runtime_set_input(handle, "A", &input);
  gettimeofday(&t2, 0);

  tvm_runtime_run(handle);
  gettimeofday(&t3, 0);

  free(params_data);
  free(input_storage);
  free(graph_data);

  float* output_storage = malloc(out_dim1 * out_dim2 * sizeof(float));
  DLTensor output;
  output.data = output_storage;
  DLContext out_ctx = {kDLCPU, 0};
  output.ctx = out_ctx;
  output.ndim = 2;
  DLDataType out_dtype = {kDLFloat, 32, 1};
  output.dtype = out_dtype;
  int64_t out_shape [2] = {out_dim1, out_dim2};
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
  int output_size = out_dim1 * out_dim2;
  for (int i = 0; i < output_size; ++i) {
    if (fabs(output_storage[i] - exp_out[i]) >= 1e-3f) {
      result = false;
      Log_Debug("got %f, expected %f\n", output_storage[i], exp_out[i]);
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

  Log_Debug("timing: %.2f ms (create), %.2f ms (set_input), %.2f ms (run), "
         "%.2f ms (get_output), %.2f ms (destroy)\n",
         (float)(t1.tv_sec-t0.tv_sec)*1000 + ((float)t1.tv_usec-(float)t0.tv_usec)/1000.f,
         (float)(t2.tv_sec-t1.tv_sec)*1000 + ((float)t2.tv_usec-(float)t1.tv_usec)/1000.f,
         (float)(t3.tv_sec-t2.tv_sec)*1000 + ((float)t3.tv_usec-(float)t2.tv_usec)/1000.f,
         (float)(t4.tv_sec-t3.tv_sec)*1000 + ((float)t4.tv_usec-(float)t3.tv_usec)/1000.f,
         (float)(t5.tv_sec-t4.tv_sec)*1000 + ((float)t5.tv_usec-(float)t4.tv_usec)/1000.f);


  float duration = (float)(t3.tv_sec-t2.tv_sec)*1000 + ((float)t3.tv_usec-(float)t2.tv_usec)/1000.f;
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
