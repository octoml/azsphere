"""Creates TVM modules for Azure Sphere."""

import argparse
import os
from tvm import relay
import tvm
from tvm import te
import logging
import json
import numpy as np
import tensorflow as tf

local = False

if local:
	TARGET = 'llvm --system-lib'
else:
    TARGET = 'llvm -target=arm-poky-linux-musleabi -mcpu=cortex-a7 --system-lib'

batch = 1
in_channel = 10
out_channel = 20
in_size = 14
kernel = 3
pad = 1
stride = 1

def build_module(opts):
    dshape = (1, 3, 224, 224)
    from mxnet.gluon.model_zoo.vision import get_model
    block = get_model('mobilenet0.25', pretrained=True)
    shape_dict = {'data': dshape}
    mod, params = relay.frontend.from_mxnet(block, shape_dict)

    # quanitization
    if (opts.quantize):
        mod = quantize(mod, params, data_aware=False)

    func = mod["main"]
    func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(
            func, target=TARGET, params=params)

    build_dir = os.path.abspath(opts.out_dir)
    if not os.path.isdir(build_dir):
        os.makedirs(build_dir)

    lib.save(os.path.join(build_dir, 'model.o'))
    with open(os.path.join(build_dir, 'graph.json'), 'w') as f_graph_json:
        f_graph_json.write(graph)
    with open(os.path.join(build_dir, 'params.bin'), 'wb') as f_params:
        f_params.write(relay.save_param_dict(params))

def build_conv2d_module(opts):
    A = relay.var('A', shape=(batch, in_channel, in_size, in_size))
    W = relay.var('W', shape=(out_channel, in_channel, kernel, kernel))
    B = relay.op.nn.nn.conv2d(A, W,
                            strides=(stride, stride),
                            padding=(pad, pad),
                            kernel_size=kernel, 
                            data_layout='NCHW', 
                            kernel_layout='OIHW',
                            out_layout='',
                            out_dtype='')

    a_data = np.random.rand(batch, in_channel, 
                            in_size, in_size).astype('float32')
    w_data = np.full((out_channel, in_channel, 
                            kernel, kernel), 1/(kernel*kernel)).astype('float32')
    func = relay.Function([A, W], B)
    params = {"W": w_data}
    graph, lib, params = relay.build_module.build(
        tvm.IRModule.from_expr(func), target=TARGET, params=params)

    build_dir = os.path.abspath(opts.out_dir)
    if not os.path.isdir(build_dir):
        os.makedirs(build_dir)
    
    ## get TVM result
    # tvm_out = run_conv2d_module(a_data, graph, lib, params, 'llvm --system-lib')
    # tf_out = tf_conv2d(a_data, w_data)
    # tf_out = np.reshape(tf_out, (batch, out_channel, in_size, in_size))
    # print(tvm_out.shape)
    # print(tf_out.shape)
    # np.testing.assert_almost_equal(2.2222, 2.2221, decimal=4)
    # np.testing.assert_almost_equal(tvm_out, tf_out, decimal=2)

    lib.save(os.path.join(build_dir, 'conv2d_model.o'))
    with open(os.path.join(build_dir, 'conv2d_graph.json'), 'w') as f_graph_json:
        f_graph_json.write(graph)
    with open(os.path.join(build_dir, 'conv2d_params.bin'), 'wb') as f_params:
        f_params.write(relay.save_param_dict(params))
    with open(os.path.join(build_dir, "conv2d_data.bin"), "wb") as fp:
        fp.write(a_data.astype(np.float32).tobytes())
    # with open(os.path.join(build_dir, "conv2d_output.bin"), "wb") as fp:
    #     fp.write(tvm_out.astype(np.float32).tobytes())

def run_conv2d_module(input, graph, lib, params, target):
    ctx = tvm.context(target, 0)
    ## create module
    module = tvm.contrib.graph_runtime.create(graph, lib, ctx)
    module.set_input('A', input)
    module.set_input(**params)
    ## run
    module.run()
    # get output
    out = module.get_output(0).asnumpy()
    return out


def tf_conv2d(a, w):
    ## Calculate output by tensor flow
    # tf_a = tf.convert_to_tensor(np.reshape(a, (batch, in_size, in_size, in_channel)))
    tf_a = a
    tf_w = tf.convert_to_tensor(np.reshape(w, (kernel, kernel, in_channel, out_channel)))
    tf_b = tf.nn.conv2d(tf_a, tf_w, strides=(stride, stride), padding="SAME", data_format='NCHW')
    b_output = tf.Session().run(tf_b)
    return b_output

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

def build_inputs(opts):
    from tvm.contrib import download
    from PIL import Image
    import numpy as np

    build_dir = os.path.abspath(opts.out_dir)

    # Download test image
    image_url = 'https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg'
    image_fn = os.path.join(build_dir, "cat.png")
    download.download(image_url, image_fn)
    image = Image.open(image_fn).resize((224, 224))

    def transform_image(image):
        image = np.array(image) - np.array([123., 117., 104.])
        image /= np.array([58.395, 57.12, 57.375])
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :]
        return image

    x = transform_image(image)
    print('x', x.shape)
    with open(os.path.join(build_dir, "cat.bin"), "wb") as fp:
        fp.write(x.astype(np.float32).tobytes())

def quantize(mod, params, data_aware):
    if data_aware:
        with relay.quantize.qconfig(calibrate_mode='kl_divergence', weight_scale='max'):
            mod = relay.quantize.quantize(mod, params, dataset=calibrate_dataset())
    else:
        with relay.quantize.qconfig(calibrate_mode='global_scale', global_scale=8.0):
            mod = relay.quantize.quantize(mod, params)
    return mod
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out-dir', default='.')
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--conv2d', action='store_true')
    opts = parser.parse_args()
    
    if opts.test:
        build_test_module(opts)
    elif opts.conv2d:
        build_conv2d_module(opts)
    else:
        build_module(opts)
        build_inputs(opts)
