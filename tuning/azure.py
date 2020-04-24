import tvm
from tvm import autotvm
from tvm import relay
from tvm import te
from topi.testing import conv2d_nchw_python
import numpy as np
from array import *
import os
import subprocess
import shutil

azure_src = [
	"bundle_static.c",
	"conv2d_network.c",
	"runtime.c",
	# "applibs_versions.h"
]

ID_OFF = 1000

# def build_model(schedule_log, batch_size=1, target=None):
# 	in_ch = 3
# 	in_size = 20
# 	out_ch = 3
# 	HH = WW = 7
# 	stride = 1
# 	pad = 3
# 	input_shape = (batch_size, in_ch, in_size, in_size)
# 	kernel_shape = (out_ch, in_ch, HH, WW)
# 	output_shape = (batch_size, out_ch, in_size, in_size)
# 	dtype = 'float32'

# 	A = relay.var('A', shape=input_shape)
# 	W = relay.var('W', shape=kernel_shape)
# 	B = relay.op.nn.nn.conv2d(A, W,
# 							strides=(stride, stride),
# 							padding=(pad, pad),
# 							kernel_size=HH, 
# 							data_layout='NCHW', 
# 							kernel_layout='OIHW',
# 							out_layout='',
# 							out_dtype='')

# 	a_data = np.random.uniform(size=(batch_size, in_ch, 
#                             in_size, in_size)).astype('float32')
# 	w_data = np.random.uniform(size=(out_ch, in_ch, 
# 							HH, WW)).astype('float32')
# 	func = relay.Function([A, W], B)
# 	params = {"W": w_data}
# 	mod = tvm.IRModule.from_expr(func)

# 	print(schedule_log)
# 	with autotvm.apply_history_best(schedule_log):
# 		with relay.build_config(opt_level=3):
# 			graph, lib, params = relay.build_module.build(
# 				mod, target=target, params=params)

# 	b_np = conv2d_nchw_python(a_data, w_data, (stride, stride), (pad, pad))
# 	return graph, lib, params, a_data, b_np

	
class AzureSphere():
	def __init__(self, key, schedule_path, target):
		self.key = key
		self.schedule_path = schedule_path
		self.target = target
		
		self.lib = None
		self.graph = None
		self.params = None 
		self.dataIn = None
		self.dataOut = None

		self.libPath = "lib_" + str(self.key).zfill(4)
		self.exportPath = None
		self.imagepackage = None
		self.output = None
		self.isBuild = False

	def build_model(self, schedule_log, batch_size=1, target=None):
		in_ch = 3
		in_size = 20
		out_ch = 3
		HH = WW = 7
		stride = 1
		pad = 3
		input_shape = (batch_size, in_ch, in_size, in_size)
		kernel_shape = (out_ch, in_ch, HH, WW)
		output_shape = (batch_size, out_ch, in_size, in_size)
		dtype = 'float32'

		A = relay.var('A', shape=input_shape)
		W = relay.var('W', shape=kernel_shape)
		B = relay.op.nn.nn.conv2d(A, W,
								strides=(stride, stride),
								padding=(pad, pad),
								kernel_size=HH, 
								data_layout='NCHW', 
								kernel_layout='OIHW',
								out_layout='',
								out_dtype='')

		a_data = np.random.uniform(size=(batch_size, in_ch, 
	                            in_size, in_size)).astype('float32')
		w_data = np.random.uniform(size=(out_ch, in_ch, 
								HH, WW)).astype('float32')
		func = relay.Function([A, W], B)
		params = {"W": w_data}
		mod = tvm.IRModule.from_expr(func)

		# print(schedule_log)
		with autotvm.apply_history_best(schedule_log):
			with relay.build_config(opt_level=3):
				graph, lib, params = relay.build_module.build(
					mod, target=target, params=params)

		b_np = conv2d_nchw_python(a_data, w_data, (stride, stride), (pad, pad))
		return graph, lib, params, a_data, b_np

	# build graph, lib, params
	def build(self):
		self.graph, self.lib, self.params, self.dataIn, self.dataOut \
		= self.build_model(schedule_log=self.schedule_path,
					  target=self.target)
		print("Model " + self.libPath + " created!")
	
	# export model and params as file
	def export(self, path):
		if not os.path.exists(path):
			os.makedirs(path)

		self.exportPath = os.path.join(path, self.libPath)
		if not os.path.exists(self.exportPath):
			os.makedirs(self.exportPath)

		build_dir = self.exportPath + '/build'
		if not os.path.exists(build_dir):
			os.makedirs(build_dir)
		
		self.lib.save(os.path.join(build_dir, 'model.o'))
		with open(os.path.join(build_dir, 'conv2d_graph.json'), 'w') as f_graph_json:
			f_graph_json.write(self.graph)
		with open(os.path.join(build_dir, 'conv2d_params.bin'), 'wb') as f_params:
			f_params.write(relay.save_param_dict(self.params))
		with open(os.path.join(build_dir, 'conv2d_data.bin'), "wb") as fp:
			fp.write(self.dataIn.astype(np.float32).tobytes())
		with open(os.path.join(build_dir, 'conv2d_output.bin'), "wb") as fp:
			fp.write(self.dataOut.astype(np.float32).tobytes())

		##generate ID
		# id = np.array(self.key + ID_OFF).astype(np.uint16)
		id = np.array(0).astype(np.uint16)
		print(id)
		with open(os.path.join(build_dir, 'id.bin'), "wb") as fp:
			id.tofile(fp)

	def dependency(self, config_path, src_path):
		if not os.path.exists(config_path) or \
			not os.path.exists(src_path):
			raise FileNotFoundError

		# config files
		cmake_file = os.path.join(config_path, "CMakeLists.txt")
		make_file = os.path.join(config_path, "Makefile")
		manifest_file = os.path.join(config_path, "app_manifest.json")
		
		shutil.copyfile(cmake_file, os.path.join(self.exportPath, "CMakeLists.txt"))
		shutil.copyfile(make_file, os.path.join(self.exportPath, "Makefile"))
		shutil.copyfile(manifest_file, os.path.join(self.exportPath, "app_manifest.json"))

		# src files
		for file in azure_src:
			shutil.copyfile(os.path.join(src_path, file),
			os.path.join(self.exportPath, file))
	
	def package(self):
		current_dir = os.path.dirname(__file__)

		# move to lib directory
		os.chdir(self.exportPath)
		process = subprocess.Popen("make " + " imagepackage",
                     shell = True,
					 stdout = subprocess.PIPE,
					 stderr=subprocess.PIPE)
		out, err = process.communicate()

		if os.path.exists(os.path.join("build", "octoml_AS.imagepackage")):
			print("Package of " + self.libPath + " created!")
		else:
			print("Package error in library " + str(self.libPath))
		
		# move to tuning directory
		os.chdir(current_dir)