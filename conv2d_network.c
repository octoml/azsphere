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
#include <unistd.h>
#include <math.h>
#include <hw/sample_hardware.h>
#include <tvm/runtime/c_runtime_api.h>

#include "bundle.h"
#include "build/conv2d_graph.json.c"
#include "build/conv2d_params.bin.c"
#include "network.h"

// Convolution
#define H       8
#define W       8
#define in_ch   3
#define out_ch  64
#define batch   1

// static volatile int fd;
// static void TerminationHandler(int signo);
static ExitCode exitCode = ExitCode_Success;
static char interface[] = "eth0";
static uint16_t serverPort = 11000;
static char serverIP[] = "192.168.0.10";
static char id[4];
static char data_file[] = "build/conv2d_data.bin";
static char output_file[] = "build/conv2d_output.bin";
static char id_file[] = "build/id.bin";

// static void TerminationHandler(int signo) {
//   // Log_Debug("Termination Handler\n");
//   GPIO_SetValue(fd, GPIO_Value_High);
//   exitCode = ExitCode_TermHandler_SigKill;
// }

int main(int argc, char **argv) {
  Log_Debug("Starting TVM Conv2d Test...\n");
  
  struct timeval t0, t1, t2, t3, t4, t5;
  gettimeofday(&t0, 0);
  char * json_data = (char *)(build_conv2d_graph_json);
  char * params_data = (char *)(build_conv2d_params_bin);
  uint64_t params_size = build_conv2d_params_bin_len;
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
  fid = Storage_OpenFileInImagePackage(id_file);
  if (fid == -1) {
    Log_Debug("Error: Openning %s failed!\n", id_file);
    goto endApp;
  }
  fs = lseek(fid, 0, SEEK_END);
  if (fs == -1) {
    Log_Debug("Error: File %s size!\n", id_file);
    goto endApp;
  }
  lseek(fid, 0, SEEK_SET);
  read(fid, &id, fs);
  close(fid);

  char msg [20];
  int len;
  len = message(id, Message_START, msg);
  send(socket , msg , len, 0);

  gettimeofday(&t0, 0);
  auto *handle = tvm_runtime_create(json_data, params_data, params_size);
  gettimeofday(&t1, 0);

  // Read data
  float input_storage[batch * in_ch * H * W];
  fid = Storage_OpenFileInImagePackage(data_file);
  if (fid == -1) {
    Log_Debug("Error: Openning %s failed!\n", data_file);
    goto endApp;
  }
  fs = lseek(fid, 0, SEEK_END);
  if (fs == -1) {
    Log_Debug("Error: File %s size!\n", data_file);
    goto endApp;
  }
  lseek(fid, 0, SEEK_SET);
  Log_Debug("%s size: %d\n", data_file, (int)fs);
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
  input.shape = shape;
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
  output.shape = out_shape;
  output.strides = NULL;
  output.byte_offset = 0;
  
  tvm_runtime_get_output(handle, 0, &output);
  gettimeofday(&t4, 0);

  tvm_runtime_destroy(handle);
  gettimeofday(&t5, 0);

// Read expected output
  float exp_out[batch * out_ch * H * W];
  fid = Storage_OpenFileInImagePackage(output_file);
  if (fid == -1) {
    Log_Debug("Error: Openning %s failed!\n", output_file);
    goto endApp;
  }
  fs = lseek(fid, 0, SEEK_END);
  if (fs == -1) {
    Log_Debug("Error: File %s size!\n", output_file);
    goto endApp;
  }
  lseek(fid, 0, SEEK_SET);
  Log_Debug("%s size: %d\n", output_file, (int)fs);
  read(fid, &exp_out, fs);
  close(fid);

  bool result = true;
  for (auto i = 0; i < batch*out_ch*H*W; ++i) {
    if (fabs(output_storage[i] - exp_out[i]) >= 1e-5f) {
      result = false;
      Log_Debug("got %f, expected %f\n", output_storage[i], exp_out[i]);
      break;
    }
  }

  len = message(id, Message_RESULT, msg);
  msg[len] = ',';
  if (result)   msg[len+1] = '1';
  else          msg[len+1] = '0';
  msg[len+2] = '\n';
  len += 3;
  send(socket , msg , len, 0);

  Log_Debug("timing: %.2f ms (create), %.2f ms (set_input), %.2f ms (run), "
         "%.2f ms (get_output), %.2f ms (destroy)\n",
         (t1.tv_sec-t0.tv_sec)*1000 + (t1.tv_usec-t0.tv_usec)/1000.f,
         (t2.tv_sec-t1.tv_sec)*1000 + (t2.tv_usec-t1.tv_usec)/1000.f,
         (t3.tv_sec-t2.tv_sec)*1000 + (t3.tv_usec-t2.tv_usec)/1000.f,
         (t4.tv_sec-t3.tv_sec)*1000 + (t4.tv_usec-t3.tv_usec)/1000.f,
         (t5.tv_sec-t4.tv_sec)*1000 + (t5.tv_usec-t4.tv_usec)/1000.f);


  uint32_t duration = (t3.tv_sec-t2.tv_sec)*1000 + (t3.tv_usec-t2.tv_usec)/1000.f;
  len = message(id, Message_TIME, msg);
  msg[len] = ',';
  len += 1;

  msg[len]    = (duration >> 24) & 0xff;
  msg[len+1]  = (duration >> 16) & 0xff;
  msg[len+2]  = (duration >> 8) & 0xff;
  msg[len+3]  = duration & 0xff;
  
  msg[len+4]  = '\n';
  len += 5;
  send(socket , msg , len, 0);
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
