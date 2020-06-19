import argparse
import os
import logging
import json
import numpy as np

import tvm
from tvm import te
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
from topi.testing import conv2d_nchw_python
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

local = False

if local:
	TARGET = 'llvm --system-lib'
else:
  TARGET = 'llvm -target=armv7e-none-eabi -mcpu=cortex-m4 -mfloat-abi=soft --system-lib'
  # TARGET = 'c'
  # -target=arm-none-eabi -mcpu=cortex-m4  -march=armv7e-m
  # TARGET = 'llvm -target=arm-poky-linux-musleabi -mcpu=cortex-a7 --system-lib'

def build_test_module(opts):
  import numpy as np

  x = relay.var('x', shape=(10, 5))
  y = relay.var('y', shape=(1, 5))
  z = relay.add(x, y)
  func = relay.Function([x, y], z)
  x_data = np.random.rand(10, 5).astype('float32')
  y_data = np.random.rand(1, 5).astype('float32')
  params = {"y": y_data}
  graph, lib, params = relay.build(
    tvm.IRModule.from_expr(func), target=TARGET, params=params)

  build_dir = os.path.abspath(opts.out_dir)
  if not os.path.isdir(build_dir):
    os.makedirs(build_dir)

  lib.save(os.path.join(build_dir, 'test_model.o'))
  with open(os.path.join(build_dir, 'test_graph.json'), 'w') as f_graph_json:
    f_graph_json.write(graph)
  with open(os.path.join(build_dir, 'test_params.bin'), 'wb') as f_params:
    f_params.write(relay.save_param_dict(params))
  with open(os.path.join(build_dir, "test_data.bin"), "wb") as fp:
    fp.write(x_data.astype(np.float32).tobytes())
  x_output = x_data + y_data
  with open(os.path.join(build_dir, "test_output.bin"), "wb") as fp:
    fp.write(x_output.astype(np.float32).tobytes())

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--out-dir', default='.')
  parser.add_argument('--test', action='store_true')
  opts = parser.parse_args()

  build_dir = os.path.abspath(opts.out_dir)
  if not os.path.isdir(build_dir):
    os.makedirs(build_dir)

  if opts.test:
    build_test_module(opts)