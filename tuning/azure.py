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

def cmake_generate(params):
	if not params['main']:
		raise RuntimeError('no main file in cmake')
	else:
		main_file = params['main']
		# main_file = 'conv2d_network.c'
		
	if params['approot_files']:
		approot_files = params['approot_files']
		approot = "\""
		for ii in range(len(approot_files)):
			if ii < len(approot_files)-1:
				approot += approot_files[ii] + ";"
			else:
				approot += approot_files[ii]
		approot += "\""

	PROJECT_NAME = '{PROJECT_NAME}'
	CFILES = '{CFILES}'
	OBJS = '{OBJS}'
	AZURE_SPHERE_MAKE_IMAGE_FILE = "{AZURE_SPHERE_MAKE_IMAGE_FILE}"
	AZURESPHERE = '{AZURESPHERE}'
	content = f'''\
CMAKE_MINIMUM_REQUIRED(VERSION 3.8)\n\
PROJECT(octoml_AS C)\n\n\
SET(CFILES {main_file} bundle_static.c)\n\
SET(OBJS build/conv2d_model.o)\n\n\
ADD_EXECUTABLE(${PROJECT_NAME} ${CFILES} ${OBJS})\n\
TARGET_LINK_LIBRARIES(${PROJECT_NAME} applibs pthread gcc_s c)\n\n\
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ~/tvm/include)\n\
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ~/tvm/src)\n\
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ~/tvm/3rdparty/dmlc-core/include)\n\
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ~/tvm/3rdparty/dlpack/include)\n\
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ../../../../include)\n\
SET(ADDITIONAL_APPROOT_INCLUDES {approot})\n\
INCLUDE(\"${AZURE_SPHERE_MAKE_IMAGE_FILE}\")\
	'''

	return content


class AzureSphere():
	def __init__(self, key, task, schedule_path, target):
		self.key = key
		self.task = task
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

	def build_model(self, task, schedule_log, batch_size=1, target=None):
		# print(task.name)
		# print(task.args)
		if task.name == 'dense_nopack.x86':
			A_param = task.args[0]
			A_shape = A_param[1]
			W_param = task.args[1]
			W_shape = W_param[1]
			dtype = task.args[3]

			A = relay.var('A', shape=A_shape, dtype=dtype)
			W = relay.var('W', shape=W_shape, dtype=dtype)
			B = relay.op.nn.nn.dense(A, W)

			a_data = np.random.uniform(size=A_shape).astype('float32')
			w_data = np.random.uniform(size=W_shape).astype('float32')
			func = relay.Function([A, W], B)
			params = {"W": w_data}
			mod = tvm.IRModule.from_expr(func)

			b_np = np.matmul(a_data, w_data.transpose())
			# print(a_data.shape)
			# print(w_data.shape)
			# print(b_np.shape)

		elif task.name == 'conv2d_NCHWc.x86':
			A_param = task.args[0]
			A_shape = A_param[1]
			W_param = task.args[1]
			W_shape = W_param[1]
			dtype = task.args[7]
			strides = task.args[4]
			pads = task.args[3]
			kernel_shape = (W_shape[2], W_shape[3])
			
			# print('padding: ', pads)
			# print('kernel: ', kernel_shape)

			A = relay.var('A', shape=A_shape, dtype=dtype)
			W = relay.var('W', shape=W_shape, dtype=dtype)
			B = relay.op.nn.nn.conv2d(A, W,
									strides=strides,
									padding=pads,
									kernel_size=kernel_shape, 
									data_layout='NCHW', 
									kernel_layout='OIHW',
									out_layout='',
									out_dtype='')

			a_data = np.random.uniform(size=A_shape).astype(dtype)
			w_data = np.random.uniform(size=W_shape).astype(dtype)
			func = relay.Function([A, W], B)
			params = {"W": w_data}
			mod = tvm.IRModule.from_expr(func)

			b_np = conv2d_nchw_python(a_data, w_data, strides, pads)
		else:
			raise RuntimeError("Task is not implemented")

		with autotvm.apply_history_best(schedule_log):
			with relay.build_config(opt_level=3):
				graph, lib, params = relay.build_module.build(
					mod, target=target, params=params)

		
		return graph, lib, params, a_data, b_np

	# build graph, lib, params
	def build(self):
		self.graph, self.lib, self.params, self.dataIn, self.dataOut \
		= self.build_model(task=self.task, schedule_log=self.schedule_path,
					  target=self.target)
		print("Model " + self.libPath + " created!")
	
	# export model and params as file
	def export(self, path):
		if not os.path.exists(path):
			os.makedirs(path)

		self.exportPath = path
		if not os.path.exists(self.exportPath):
			os.makedirs(self.exportPath)
		
		build_dir = self.exportPath + '/build'
		if not os.path.exists(build_dir):
			os.makedirs(build_dir)
		
		self.lib.save(os.path.join(build_dir, 'conv2d_model.o'))
		with open(os.path.join(build_dir, 'conv2d_graph.json'), 'w') as f_graph_json:
			f_graph_json.write(self.graph)

		# print(type(self.graph))
		with open(os.path.join(build_dir, 'conv2d_graph.bin'), 'wb') as f_graph:
			f_graph.write(bytes(self.graph, 'utf-8'))


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

	def dependency(self, config_path, src_path, params=None):
		if not os.path.exists(config_path) or \
			not os.path.exists(src_path):
			raise FileNotFoundError

		# config files
		cmake_file = os.path.join(config_path, "CMakeLists.txt")
		if params:
			cmake_data = cmake_generate(params)
			with open(os.path.join(self.exportPath, "CMakeLists.txt"), 'w+') as f:
				f.write(cmake_data)
		else:
			shutil.copyfile(cmake_file, os.path.join(self.exportPath, "CMakeLists.txt"))

		make_file = os.path.join(config_path, "Makefile")
		manifest_file = os.path.join(config_path, "app_manifest.json")
		shutil.copyfile(make_file, os.path.join(self.exportPath, "Makefile"))
		shutil.copyfile(manifest_file, os.path.join(self.exportPath, "app_manifest.json"))

		#TODO: make previous setup compatible to this
		# src files
		if params['src_files']:
			for file in params['src_files']:
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
		
		process.kill()
		# move to tuning directory
		os.chdir(current_dir)