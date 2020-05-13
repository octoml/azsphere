####
## model from: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/micro_speech
###
import tvm
from tvm import te
from tvm import relay

# os and numpy
import numpy as np
import os.path

# Tensorflow imports
import tensorflow as tf
# tf.contrib.resampler

try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing

# Base location for model related files.
repo_base = 'https://github.com/mehrdadhe/ML-KWS-for-MCU/tree/master/Pretrained_models'

# Test image
# img_name = 'elephant-299.jpg'
# image_url = os.path.join(repo_base, img_name)

######################################################################
model_file_name = 'model.pb'
model_name = 'micro_speech'
model_dir = os.path.join(repo_base, 'DS_CNN')
model_url = os.path.join(model_dir, model_file_name)

# Image label map
# map_proto = 'imagenet_2012_challenge_label_map_proto.pbtxt'
# map_proto_url = os.path.join(repo_base, map_proto)

# Human readable text for labels
label_map = 'labels.txt'
label_map_url = os.path.join(repo_base, label_map)

# Target settings
# Use these commented settings to build for cuda.
#target = 'cuda'
#target_host = 'llvm'
#layout = "NCHW"
#ctx = tvm.gpu(0)
# target = 'llvm --system-lib'
# target_host = 'llvm --system-lib'
target = 'llvm -target=arm-poky-linux-musleabi -mcpu=cortex-a7 --system-lib'
layout = None
# ctx = tvm.cpu(0)

######################################################################
# Download required files
# -----------------------
# Download files listed above.
from tvm.contrib.download import download_testdata

# img_path = download_testdata(image_url, img_name, module='data')
# model_path = download_testdata(model_url, model_name, module=['tf', 'DS_CNN_S'])
model_path = os.path.join('micro_speech_model', model_file_name)
# map_proto_path = download_testdata(map_proto_url, map_proto, module='data')
label_path = download_testdata(label_map_url, label_map, module='data')

build_dir = 'build'

if not os.path.exists(build_dir):
    os.makedirs(build_dir)
######################################################################
# Import model

# tf.load_op_library('/home/parallels/.local/lib/python3.6/site-packages/tensorflow/contrib/resampler/python/ops/_resampler_ops.so')

print("model path: ", model_path)
with tf_compat_v1.gfile.GFile(model_path, 'rb') as f:
# with tf.gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf_compat_v1.GraphDef()
    graph_def.ParseFromString(f.read())
    # graph = tf.import_graph_def(graph_def, name='')
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)

    # Add shapes to the graph.
    # with tf_compat_v1.Session() as sess:
    #     graph_def = tf_testing.AddShapesToGraphDef(sess, 'softmax')
    


  # with (filename, 'rb') as f:
  #   graph_def = tf.GraphDef()
  #   graph_def.ParseFromString(f.read())
  #   tf.import_graph_def(graph_def, name='')

# from PIL import Image
# image = Image.open(img_path).resize((299, 299))

# x = np.array(image)

with open(os.path.join(build_dir, f'{model_name}_graph_original.log'), 'w') as orig_file:
    orig_file.write(str(graph_def))

##remove pre-processing nodes and fix begining
nodes = []
##add first op
#TODO: change these based on the output of Mfcc function
input_dim0 = 1
input_dim1 = 2
new_input = graph_def.node.add()
new_input.op = 'Placeholder'  # eg: 'Const', 'Placeholder', 'Add' etc
new_input.name = 'decoded_sample_data'
new_input.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(
        type=dtypes.float32.as_datatype_enum))
new_input.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            [input_dim0, input_dim1], dtypes.float32, [input_dim0, input_dim1])))
nodes.append(new_input)

for ii, node in enumerate(graph_def.node, start=0):
    if node.op == 'DecodeWav' or \
       node.op == 'Placeholder':
       pass
    else:
        nodes.append(node) 

new_graph = tf_compat_v1.GraphDef()
new_graph.node.extend(nodes)

with open(os.path.join(build_dir, f'{model_name}_graph_new.log'), 'w') as new_graph_log:
    new_graph_log.write(str(new_graph))


shape_dict = {'decoded_sample_data': (1, 10, 10, 10)}
dtype_dict = {'DecodeJpeg/contents': 'uint8'}
mod, params = relay.frontend.from_tensorflow(new_graph,
                                             layout=layout,
                                             shape=shape_dict)

with open(os.path.join(build_dir, f'{model_name}_mod.log'), 'w') as mod_file:
    mod_file.write(str(mod))

# print("Tensorflow protobuf imported to relay frontend.")

# with relay.build_config(opt_level=3):
#     graph, lib, params = relay.build(mod,
#                                      target=target,
#                                      params=params)
# model_exp_name = 'arm_keyword'

# #save model, graph, params
# lib.save(os.path.join(build_dir, f'{model_exp_name}_model.o'))
# with open(os.path.join(build_dir, f'{model_exp_name}_graph.bin'), 'wb') as f_graph:
#     f_graph.write(bytes(graph, 'utf-8'))
# with open(os.path.join(build_dir, f'{model_exp_name}_graph.json'), 'w') as f_graph_json:
#     f_graph_json.write(graph)
# with open(os.path.join(build_dir, f'{model_exp_name}_params.bin'), 'wb') as f_params:
#     f_params.write(relay.save_param_dict(params))

# from tvm.contrib import graph_runtime
# dtype = 'uint8'
# m = graph_runtime.create(graph, lib, ctx)
# # set inputs
# m.set_input('DecodeJpeg/contents', tvm.nd.array(x.astype(dtype)))
# m.set_input(**params)
# # execute
# m.run()
# # get outputs
# tvm_output = m.get_output(0, tvm.nd.empty(((1, 1008)), 'float32'))


# predictions = tvm_output.asnumpy()
# predictions = np.squeeze(predictions)

# Creates node ID --> English string lookup.
# node_lookup = tf_testing.NodeLookup(label_lookup_path=map_proto_path,
                                    # uid_lookup_path=label_path)

# Print top 5 predictions from TVM output.
# top_k = predictions.argsort()[-5:][::-1]
# for node_id in top_k:
#     human_string = node_lookup.id_to_string(node_id)
#     score = predictions[node_id]
#     print('%s (score = %.5f)' % (human_string, score))

######################################################################
# Inference on tensorflow
# -----------------------
# Run the corresponding model on tensorflow

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf_compat_v1.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

def run_inference_on_image(image):
    """Runs inference on an image.

    Parameters
    ----------
    image: String
        Image file name.

    Returns
    -------
        Nothing
    """
    if not tf_compat_v1.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf_compat_v1.gfile.GFile(image, 'rb').read()

    # Creates graph from saved GraphDef.
    create_graph()

    with tf_compat_v1.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})

        predictions = np.squeeze(predictions)

        # Creates node ID --> English string lookup.
        node_lookup = tf_testing.NodeLookup(label_lookup_path=map_proto_path,
                                            uid_lookup_path=label_path)

        # Print top 5 predictions from tensorflow.
        top_k = predictions.argsort()[-5:][::-1]
        print ("===== TENSORFLOW RESULTS =======")
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))

# run_inference_on_image(img_path)
