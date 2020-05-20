"""Creates TVM modules for Azure Sphere."""
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
    TARGET = 'llvm -target=arm-poky-linux-musleabi -mcpu=cortex-a7 --system-lib'
    # TARGET = 'llvm -device=arm_cpu -target=arm-poky-linux-musleabi -mcpu=cortex-a7 --system-lib'

BATCH = 1
IN_CHANNEL = 3
OUT_CHANNEL = 3
IN_SIZE = 20
KERNEL = 7
PAD = 3
STRIDE = 1

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

def build_cifar(opts, model_name):
    from tuning.model.cifar10_relay import get_cifar_relay
    from tuning.model.cifar10_arm import get_cifar_keras, gen_custom_cifar_keras

    if model_name == 'cifar-10':
        # cifar_path = 'tuning/model/saved_models/cifar10_ch8_best.h5'
        # model_input_name = 'conv2d_1_input'
        model_input_name = 'cifar10_arm_input'
        # cifar_path = 'tuning/model/saved_models/cifar10_arm_best.h5'
        shape_dict = {model_input_name: (1, 3, 32, 32)}
        # model = get_cifar_keras(cifar_path, shape_dict)

        model = gen_custom_cifar_keras(shape_dict)
        mod, params = tvm.relay.frontend.from_keras(model, shape_dict)

        print("Compile...")
        if opts.tuned:
            print("INFO: Tuned model!")
            with autotvm.apply_history_best(os.path.join('tuning', 'cifar_arm_footprint_min.txt')):
                if opts.quantize:
                    with relay.quantize.qconfig(calibrate_mode='global_scale', global_scale=8.0):
                        mod = relay.quantize.quantize(mod, params)
                        print('INFO: Quantized!')

                with relay.build_config(opt_level=3):
                    graph, lib, params = relay.build_module.build(
                        mod, target=TARGET, params=params)
        else:
            print("INFO: No Tuning!")
            with relay.build_config(opt_level=3):
                    graph, lib, params = relay.build_module.build(
                        mod, target=TARGET, params=params)

    elif model_name == 'cifar-10-relay':
        mod, params = get_cifar_relay()
        print('type:            ', type(mod))
        print(mod.get_global_type_var)
        shape_dict = {'data': (1, 3, 32, 32)}

        print("Compile...")
        if opts.tuned:
            print("INFO: Tuned model!")
            with autotvm.apply_history_best(os.path.join('tuning', 'cifar_relay_footprint_min.txt')):
                if opts.quantize:
                    with relay.quantize.qconfig(calibrate_mode='global_scale', global_scale=8.0):
                        mod = relay.quantize.quantize(mod, params)
                        print('INFO: Quantized!')

                with relay.build_config(opt_level=3):
                    graph, lib, params = relay.build_module.build(
                        mod, target=TARGET, params=params)
        else:
            print("INFO: No Tuning!")
            with relay.build_config(opt_level=3):
                    graph, lib, params = relay.build_module.build(
                        mod, target=TARGET, params=params)
    else:
        raise ValueError('Wrong model name!')

    #save model, graph, params
    lib.save(os.path.join(build_dir, 'cifar_model.o'))
    with open(os.path.join(build_dir, 'cifar_graph.bin'), 'wb') as f_graph:
        f_graph.write(bytes(graph, 'utf-8'))
    with open(os.path.join(build_dir, 'cifar_graph.json'), 'w') as f_graph_json:
        f_graph_json.write(graph)
    with open(os.path.join(build_dir, 'cifar_params.bin'), 'wb') as f_params:
        f_params.write(relay.save_param_dict(params))

    #create input and result
    if model_name == 'cifar-10':
        import keras
        from keras.datasets import cifar10
        # from keras.models import load_model

        num_classes = 10
        # model = load_model(cifar_path)
        (_, _), (x_test, y_test) = cifar10.load_data()

        x_test = x_test.astype('float32')
        x_test /= 255
        y_test = keras.utils.to_categorical(y_test, num_classes)

        test_x_sample = x_test[0:1,:,:,:]
        test_y_sample = y_test[0:1,:]
        print('x_test_sample shape:', test_x_sample.shape)
        print('y_test_sample shape:', test_y_sample.shape)
        scores = model.evaluate(test_x_sample, test_y_sample, verbose=1)
        keras_predict = model.predict(test_x_sample)
        print(keras_predict)
        
        ## get TVM result on local machine
        mod, params = relay.frontend.from_keras(model, shape_dict)
        local_target = 'llvm --system-lib'
        
        if opts.quantize:
            with relay.quantize.qconfig(calibrate_mode='global_scale', global_scale=8.0):
                mod = relay.quantize.quantize(mod, params)

        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=local_target, params=params)

        ctx = tvm.context(local_target, 0)
        ## create module
        module = tvm.contrib.graph_runtime.create(graph, lib, ctx)
        tvm_sample = test_x_sample.transpose([0, 3, 1, 2])
        # print("tvm_sample shape: ", tvm_sample.shape)
        module.set_input(model_input_name, tvm_sample)
        module.set_input(**params)
        ## run
        module.run()
        # get output
        tvm_out = module.get_output(0).asnumpy()
        
        print("TVM Output: " + str(tvm_out.shape))
        print("Keras Output: " + str(keras_predict.shape))
        if not opts.quantize:
            np.testing.assert_allclose(tvm_out, keras_predict, rtol=1e-2)
    elif model_name == 'cifar-10-relay':
        tvm_sample = np.array([1])
        tvm_out = np.array([1])
    else:
        raise ValueError('Wrong model name!')

    # save data and output
    with open(os.path.join(build_dir, "cifar_data.bin"), "wb") as fp:
        fp.write(tvm_sample.astype(np.float32).tobytes())
    with open(os.path.join(build_dir, "cifar_output.bin"), "wb") as fp:
        fp.write(tvm_out.astype(np.float32).tobytes())

    generate_id()

def build_keyword_model(opts):
    from tuning.model.keyword_spotting import get_module, prepare_input

    model_input_name = 'Mfcc'
    shape_dict = {model_input_name: (1, 49, 10)}

    mod = get_module('tuning/model/keyword_model/module_gs_4.0_conv_notquantized.pickle')
    print(mod)
    
    print("Compile...")
    if opts.tuned:
        print("INFO: Model tunning for runtime!")
        with autotvm.apply_history_best(os.path.join('tuning', 'keyword_runtime_min.txt')):
            with relay.build_config(opt_level=3):
                graph, lib, out_params = relay.build_module.build(
                    mod, target=TARGET)
    elif opts.footprint:
        history_file = 'keyword_footprint_min_1.txt'
        print(f'INFO: Model tuning for footprint with file {history_file}!')
        with autotvm.apply_history_best(os.path.join('tuning', history_file)):
            with relay.build_config(opt_level=3):
                graph, lib, out_params = relay.build_module.build(
                    mod, target=TARGET)  
    else:
        print("INFO: No Tuning!")
        with relay.build_config(opt_level=3):
                graph, lib, out_params = relay.build_module.build(
                    mod, target=TARGET)

    #save model, graph, params
    model_name = 'keyword'
    lib.save(os.path.join(build_dir, f'{model_name}_model.o'))
    with open(os.path.join(build_dir, f'{model_name}_graph.bin'), 'wb') as f_graph:
        f_graph.write(bytes(graph, 'utf-8'))
    with open(os.path.join(build_dir, f'{model_name}_graph.json'), 'w') as f_graph_json:
        f_graph_json.write(graph)
    with open(os.path.join(build_dir, f'{model_name}_params.bin'), 'wb') as f_params:
        f_params.write(relay.save_param_dict(out_params))

    #create input and result
    local_target = 'llvm --system-lib'
    with relay.build_config(opt_level=3):
        graph_test, lib_test, params_test = relay.build_module.build(
            mod, target=local_target)

    with open('build/graph.log', 'w') as f:
        f.write(str(graph))

    input_data = prepare_input('tuning/model/keyword_model/silence.wav')
    ctx = tvm.context(local_target, 0)
    m = tvm.contrib.graph_runtime.create(graph_test, lib_test, ctx)
    m.set_input('Mfcc', input_data)
    m.set_input(**params_test)
    m.run()
    predictions = m.get_output(0, tvm.nd.empty(((1, 12)), 'float32')).asnumpy()
    predictions = predictions[0]

    # save data and output
    with open(os.path.join(build_dir, f'{model_name}_data.bin'), "wb") as fp:
        fp.write(input_data.astype(np.float32).tobytes())
    with open(os.path.join(build_dir, f'{model_name}_output.bin'), "wb") as fp:
        fp.write(predictions.astype(np.float32).tobytes())

    generate_id()

def build_conv2d_module(opts):
    A = relay.var('A', shape=(BATCH, IN_CHANNEL, IN_SIZE, IN_SIZE))
    W = relay.var('W', shape=(OUT_CHANNEL, IN_CHANNEL, KERNEL, KERNEL))
    B = relay.op.nn.nn.conv2d(A, W,
                            strides=(STRIDE, STRIDE),
                            padding=(PAD, PAD),
                            kernel_size=KERNEL, 
                            data_layout='NCHW', 
                            kernel_layout='OIHW',
                            out_layout='',
                            out_dtype='')

    a_data = np.random.uniform(size=(BATCH, IN_CHANNEL, 
                            IN_SIZE, IN_SIZE)).astype('float32')
    w_data = np.random.uniform(size=(OUT_CHANNEL, IN_CHANNEL, 
                            KERNEL, KERNEL)).astype('float32')
    func = relay.Function([A, W], B)
    params = {"W": w_data}
    graph, lib, params = relay.build_module.build(
        tvm.IRModule.from_expr(func), target=TARGET, params=params)

    build_dir = os.path.abspath(opts.out_dir)
    if not os.path.isdir(build_dir):
        os.makedirs(build_dir)
    
    lib.save(os.path.join(build_dir, 'conv2d_model.o'))
    with open(os.path.join(build_dir, 'conv2d_graph.json'), 'w') as f_graph_json:
        f_graph_json.write(graph)
    with open(os.path.join(build_dir, 'conv2d_params.bin'), 'wb') as f_params:
        f_params.write(relay.save_param_dict(params))
    with open(os.path.join(build_dir, "conv2d_data.bin"), "wb") as fp:
        fp.write(a_data.astype(np.float32).tobytes())
    
    ## get TVM result on local machine
    params = {"W": w_data}
    local_target = 'llvm --system-lib'
    graph, lib, params = relay.build_module.build(
        tvm.IRModule.from_expr(func), target=local_target, params=params)
    tvm_out = run_conv2d_module(a_data, graph, lib, params, target=local_target)
    b_np = conv2d_nchw_python(a_data, w_data, (STRIDE, STRIDE), (PAD, PAD))
    print("TVM Output: " + str(tvm_out.shape))
    print("Numpy Output: " + str(b_np.shape))
    np.testing.assert_allclose(b_np, tvm_out, rtol=1e-2)
    with open(os.path.join(build_dir, "conv2d_output.bin"), "wb") as fp:
        fp.write(tvm_out.astype(np.float32).tobytes())

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

def get_conv2d(batch_size):
    in_ch = 3
    H = 224
    W = 224
    out_ch = 512
    HH = WW = 7
    input_shape = (batch_size, in_ch, H, W)
    kernel_shape = (out_ch, in_ch, HH, WW)
    output_shape = (batch_size, out_ch, H, W)
    dtype = 'float32'

    A = relay.var('A', shape=input_shape)
    W = relay.var('W', shape=kernel_shape)
    B = relay.op.nn.nn.conv2d(A, W,
                            strides=(STRIDE, STRIDE),
                            padding=(PAD, PAD),
                            kernel_size=HH, 
                            data_layout='NCHW', 
                            kernel_layout='OIHW',
                            out_layout='',
                            out_dtype='')

    w_data = np.random.uniform(size=(out_ch, in_ch, 
                            HH, WW)).astype('float32')
    func = relay.Function([A, W], B)
    params = {"W": w_data}
    mod = tvm.IRModule.from_expr(func)
    return mod, params, input_shape, output_shape

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)
    dtype = 'float32'

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet':
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size)
    elif name == 'squeezenet_v1.1':
        mod, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape

def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        # import pdb; pdb.set_trace()
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        print(tmp_log_file)
        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=tsk_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)
                       ])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

def tune_and_evaluate(opts, tuning_opt):
    # target = tvm.target.create('llvm -device=arm_cpu -target=aarch64-linux-gnu')
    target = TARGET + ' -device_name=azure-sphere'
    print(target)
    # extract workloads from relay program
    print("Extract tasks...")
    # mod, params, input_shape, _ = get_network('resnet-18', batch_size=1)
    mod, params, input_shape, _ = get_conv2d(batch_size=1)
    # pdb.set_trace()
    tasks = autotvm.task.extract_from_program(mod['main'], target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"),))

    # with relay.build_config(opt_level=3):
    #     graph, lib, params = relay.build_module.build(
    #         mod, target=target, params=params)

    # build_dir = os.path.abspath(opts.out_dir)
    # if not os.path.isdir(build_dir):
    #     os.makedirs(build_dir)

    # lib.save(os.path.join(build_dir, 'res_conv_model.o'))
    # with open(os.path.join(build_dir, 'res_conv_graph.json'), 'w') as f_graph_json:
    #     f_graph_json.write(graph)
    # with open(os.path.join(build_dir, 'res_conv_params.bin'), 'wb') as f_params:
    #     f_params.write(relay.save_param_dict(params))
    # with open(os.path.join(build_dir, "test_data.bin"), "wb") as fp:
    #     fp.write(x_data.astype(np.float32).tobytes())

    # # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # # compile kernels with history best records
    # with autotvm.apply_history_best(log_file):
    #     print("Compile...")
    #     with relay.build_config(opt_level=3):
    #         graph, lib, params = relay.build_module.build(
    #             mod, target=target, params=params)

    #     # export library
    #     tmp = tempdir()
    #     if use_android:
    #         from tvm.contrib import ndk
    #         filename = "net.so"
    #         lib.export_library(tmp.relpath(filename), ndk.create_shared)
    #     else:
    #         filename = "net.tar"
    #         lib.export_library(tmp.relpath(filename))

    #     # upload module to device
    #     print("Upload...")
    #     remote = autotvm.measure.request_remote(device_key, '0.0.0.0', 9190,
    #                                             timeout=10000)
    #     remote.upload(tmp.relpath(filename))
    #     rlib = remote.load_module(filename)

    #     # upload parameters to device
    #     ctx = remote.context(str(target), 0)
    #     module = runtime.create(graph, rlib, ctx)
    #     data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    #     module.set_input('data', data_tvm)
    #     module.set_input(**params)

    #     # evaluate
    #     print("Evaluate inference time cost...")
    #     ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=10)
    #     prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    #     print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
    #           (np.mean(prof_res), np.std(prof_res)))

def generate_id(id=0):
    id = np.array(id).astype(np.uint16)
    with open(os.path.join(build_dir, 'id.bin'), "wb") as fp:
        id.tofile(fp)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out-dir', default='.')
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--conv2d', action='store_true')
    parser.add_argument('--conv2dauto', action='store_true')
    parser.add_argument('--x86', action='store_true')
    parser.add_argument('--cifar', action='store_true')
    parser.add_argument('--keyword', action='store_true')
    parser.add_argument('--tuned', action='store_true')
    parser.add_argument('--footprint', action='store_true')
    opts = parser.parse_args()
    
    build_dir = os.path.abspath(opts.out_dir)
    if not os.path.isdir(build_dir):
        os.makedirs(build_dir)

    if opts.test:
        build_test_module(opts)
    elif opts.conv2d:
        build_conv2d_module(opts)
    elif opts.conv2dauto:
        if opts.x86:
            log_file = "%s.log" % ('conv2d-network')
            tuning_option = {
                'log_filename': log_file,
                'tuner': 'random',
                'n_trial': 1500,
                'early_stopping': None,
                'measure_option': autotvm.measure_option(
                    builder=autotvm.LocalBuilder(),
                    runner=autotvm.LocalRunner(
                        number=10, repeat=1,
                        min_repeat_ms=1000
                    ),
                ),
                'use_transfer_learning' : False,
            }
        else:
            # Also replace this with the device key in your tracker
            device_key = 'AzureSphere'
            # Set this to True if you use android phone
            use_android = False
            #### TUNING OPTION ####
            log_file = "%s.%s.log" % (device_key, 'conv2d-network')
            tuning_option = {
                'log_filename': log_file,
                'tuner': 'xgb',
                'n_trial': 1500,
                'early_stopping': 800,
                'measure_option': autotvm.measure_option(
                    builder=autotvm.LocalBuilder(),
                    # runner=autotvm.RPCRunner(
                    runner=autotvm.DebugRunner(
                        # gdb="/opt/azurespheresdk/Sysroots/4/tools/sysroots/x86_64-pokysdk-linux/usr/bin/arm-poky-linux-musleabi/arm-poky-linux-musleabi-gdb",
                        gdb="gdb",
                        key=device_key, host='0.0.0.0', port=9190,
                        number=5,
                        timeout=10,
                    ),
                ),
            }
        tune_and_evaluate(opts, tuning_option)
    elif opts.cifar:
        build_cifar(opts, model_name='cifar-10')
        # build_cifar(opts, model_name='cifar-10-relay')
    elif opts.keyword:
        build_keyword_model(opts)
    else:
        build_module(opts)
        build_inputs(opts)
