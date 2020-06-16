import argparse
import tvm
from tvm import te
from tvm import relay

import numpy as np
import os.path

# Tensorflow imports
import tensorflow as tf
try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing
import pickle
import sys

DEBUG_LOG = False
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
OPTS = None
# Base location for model related files.
repo_base = 'https://github.com/ARM-software/ML-KWS-for-MCU/raw/master/Pretrained_models'
model_name = 'DS_CNN_S'
model_file_name = f'{model_name}.pb'
model_dir = os.path.join(repo_base, 'DS_CNN')
model_url = os.path.join(model_dir, model_file_name)
label_name = 'labels.txt'
label_url = os.path.join(repo_base, label_name)

def export_module(opts):
    # Target settings
    layout = "NCHW"

    # Download required files
    from tvm.contrib.download import download_testdata
    model_path = download_testdata(model_url, model_file_name, module=['tf', 'keyword_spotting'])
    label_path = download_testdata(label_url, label_name, module=['data'])

    # Import model
    with tf_compat_v1.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        with tf_compat_v1.Session() as sess:
            graph_def = tf_testing.AddShapesToGraphDef(sess, 'labels_softmax')

    build_dir = opts.out_dir
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    ##save original TF graph
    if DEBUG_LOG:
        with open(os.path.join(build_dir, f'{model_name}_graph_original.log'), 'w') as orig_file:
            orig_file.write(str(graph_def))

    ##remove pre-processing nodes and fix begining
    nodes = []
    ##add first op
    input_dim0 = 1
    input_dim1 = 49
    input_dim2 = 10
    new_input = graph_def.node.add()
    new_input.op = 'Placeholder'
    new_input.name = 'Mfcc'
    new_input.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(
            type=dtypes.float32.as_datatype_enum))
                
    nodes.append(new_input)

    removed_count = 0
    for ii, node in enumerate(graph_def.node, start=0):
        if node.op == 'DecodeWav' \
        or node.op == 'AudioSpectrogram' \
        or node.op == 'Mfcc' \
        or node.op == 'Placeholder' \
        or node.op == 'wav_data':
            removed_count += 1
            pass
        else:
            nodes.append(node) 
    print(f'NUM of layers removed: {removed_count}')

    new_graph = tf_compat_v1.GraphDef()
    new_graph.node.extend(nodes)
    ##log new graph
    if DEBUG_LOG:
        with open(os.path.join(build_dir, f'{model_name}_graph_new.log'), 'w') as new_graph_log:
            new_graph_log.write(str(new_graph))

    ##get mod and params with new graph
    shape_dict = {'Mfcc': (1, 49, 10)}
    mod, params = relay.frontend.from_tensorflow(new_graph,
                                                layout=layout,
                                                shape=shape_dict)

    if DEBUG_LOG:
        with open(os.path.join(build_dir, f'{model_name}_mod.log'), 'w') as mod_file:
            mod_file.write(str(mod))
        with open(os.path.join(build_dir, f'{model_name}_param.log'), 'w') as param_log:
            param_log.write(str(params))

    #quantization
    if opts.quantize:
        if not opts.global_scale:
            raise RuntimeError('Global Scale is not valid!')
        global_scale = float(opts.global_scale)
        print('INFO: Quantizing...')
        print(f'INFO: Global Scale: {global_scale}')
        with relay.quantize.qconfig(calibrate_mode='global_scale', 
                                    global_scale=global_scale,
                                    skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod, params)

        if DEBUG_LOG:
            with open(os.path.join(build_dir, f'{model_name}_mod_quantized.log'), 'w') as mod_log:
                mod_log.write(str(mod))

    #save module
    if opts.quantize:
        file_path = f'{build_dir}/module_gs_{global_scale}.pickle'
        with open(file_path, 'wb') as h1:
            pickle.dump(mod, h1, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'INFO: {file_path} saved!')
        with open(f'{build_dir}/module_gs_{global_scale}.txt', 'w') as f:
            f.write(mod.astext())
    else:
        file_path = f'{build_dir}/module.pickle'
        with open(file_path, 'wb') as h1:
            pickle.dump(mod, h1, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'INFO: {file_path} saved!')
        param_path = f'{build_dir}/params.bin'
        with open(param_path, 'wb') as f_params:
            f_params.write(relay.save_param_dict(params))
            print(f'INFO: {param_path} saved!')
        with open(f'{build_dir}/module.txt', 'w') as f:
            f.write(mod.astext())
    return mod, params

def prepare_input(filename):
    from tensorflow.contrib.framework.python.ops import audio_ops
    from tensorflow.python.ops import io_ops

    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_filename_placeholder)
        wav_decoder = tf.audio.decode_wav(wav_loader,
                            desired_channels=1,
                            desired_samples=16000,
                            name='decoded_sample_data')

        spectrum = audio_ops.audio_spectrogram(input=wav_decoder[0],
                                            window_size=640,
                                            stride=320,
                                            magnitude_squared=True,
                                            name='AudioSpectrogram')
        final = audio_ops.mfcc(spectrogram=spectrum, 
                               sample_rate=wav_decoder[1], 
                               upper_frequency_limit=4000.0, 
                               lower_frequency_limit=20.0, 
                               filterbank_channel_count=40, 
                               dct_coefficient_count=10, 
                               name='Mfcc')

        data = sess.run(final,
        feed_dict={wav_filename_placeholder: filename})
        print(f'Data shape: {data.shape}')

    return data

def load_labels():
    from tvm.contrib.download import download_testdata
    label_path = download_testdata(label_url, label_name, module=['data'])
    """Read in labels, one label per line."""
    return [line.rstrip() for line in tf.io.gfile.GFile(label_path)]

def build(opts, mod=None, params=None, target=None):
    if not target:
        raise RuntimeError('target is not valid!')

    build_dir = opts.out_dir
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    #build relay
    if opts.quantize:
        with relay.build_config(opt_level=3):
            graph, lib, out_params = relay.build(mod,
                                            target=target)
    else:
        with relay.build_config(opt_level=3):
            graph, lib, out_params = relay.build(mod,
                                            target=target,
                                            params=params)
    return lib, graph, out_params

def test_sample(target, filepath):
    input_data = prepare_input(filepath)
    if OPTS.module:
        with open(OPTS.module, 'rb') as handle:
            module = pickle.load(handle)
    else:
        module = None

    lib, graph, out_params = build(OPTS, mod=module, params=None, target='llvm --system-lib')
    ctx = tvm.context(target, 0)
    m = tvm.contrib.graph_runtime.create(graph, lib, ctx)
    m.set_input('Mfcc', input_data)
    m.set_input(**out_params)
    m.run()
    predictions = m.get_output(0, tvm.nd.empty(((1, 12)), 'float32')).asnumpy()
    predictions = predictions[0]

    labels = load_labels()

    num_top_predictions = 12
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for item in top_k:
        print('{:>12}\t{:>15}'.format(labels[item], predictions[item]))
    
def test_accuracy(opts, target):
    with open(opts.module, 'rb') as handle:
        mod = pickle.load(handle)

    params_data = None
    if not opts.quantize:
        with open(opts.params, 'rb') as f_params:
            params_data = relay.load_param_dict(f_params.read())
    
    lib, graph, out_params = build(opts, mod=mod, 
                                   params=params_data, target=target)
    ctx = tvm.context(target, 0)
    m = tvm.contrib.graph_runtime.create(graph, lib, ctx)

    #get test data
    num_of_samples = int(opts.test)
    test_data, test_label = get_dataset(num_of_samples)
    print(f'INFO: testing {num_of_samples} samples')

    #eval data
    corrects = 0
    count = 0
    for test, label in zip(test_data, test_label):
        count += 1
        input_data = test.reshape((1, 49, 10))
        m.set_input('Mfcc', input_data)
        m.set_input(**out_params)
        m.run()
        predictions = m.get_output(0, tvm.nd.empty(((1, 12)), 'float32')).asnumpy()
        predictions = predictions[0]
        exp_ind = np.argmax(label)
        pred_ind = np.argmax(predictions)
        if pred_ind == exp_ind:
            corrects += 1

    acc = corrects/(num_of_samples * 1.0)
    print(f'Accuracy for {num_of_samples} samples: {acc}')
    return acc

def get_module(filename):
    with open(filename, 'rb') as handle:
        mod = pickle.load(handle)
    return mod

def get_dataset(num_of_samples):
    import input_data
    import models

    wanted_words = 'yes,no,up,down,left,right,on,off,stop,go'
    model_settings = models.prepare_model_settings(
        label_count=len(input_data.prepare_words_list(wanted_words.split(','))),
        sample_rate=16000,
        clip_duration_ms=1000,
        window_size_ms=40.0,
        window_stride_ms=20.0,
        dct_coefficient_count=10
      )
    audio_processor = input_data.AudioProcessor(
        data_url='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
        data_dir='/tmp/speech_dataset/',
        silence_percentage=10.0,
        unknown_percentage=10.0,
        wanted_words=wanted_words.split(','),
        validation_percentage=10,
        testing_percentage=10,
        model_settings=model_settings
        )

    print(audio_processor)
    set_size = audio_processor.set_size('testing')
    batch_size = num_of_samples
    sess = tf.InteractiveSession()

    tf.logging.info('set_size=%d', set_size)
    total_accuracy = 0
    total_conf_matrix = None
    data, label = audio_processor.get_data(
        batch_size, 0, model_settings, 0.0, 0.0, 0, 'testing', sess)
    
    return data, label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out-dir', default='build', help="Output directory")
    parser.add_argument('--export', action='store_true', help="Export model")
    parser.add_argument('--quantize', action='store_true', help="Quantize KWS model")
    parser.add_argument('--global-scale', default=None, help="Global scale parameter for Relay quantization")
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--target', default='llvm --system-lib')
    parser.add_argument('--test', action=None, help="Test accuracy of the model with num of samples")
    parser.add_argument('--module', default=None, help="Path to the Relay module pickle file")
    parser.add_argument('--params', default=None, help="Path to the Relay params file")
    parser.add_argument('--debug', action='store_true', help="Save debugging files")
    parser.add_argument('--wav', default='', help="WAV test file")

    OPTS = parser.parse_args()
    DEBUG_LOG = OPTS.debug

    if OPTS.export:
        mod, params = export_module(OPTS)
        
    if OPTS.build:
        with open(OPTS.module, 'rb') as handle:
            mod = pickle.load(handle)
        build(OPTS, mod=mod, target='llvm --system-lib')

    if OPTS.test:
        test_accuracy(OPTS, target='llvm --system-lib')
    
    if OPTS.wav:
        test_sample(target='llvm --system-lib', filepath=OPTS.wav)