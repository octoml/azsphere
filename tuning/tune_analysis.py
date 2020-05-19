import struct
import argparse
import json
import matplotlib.pyplot as plt
import statistics
import numpy as np 
import os
import csv

KERNEL = 7
CI = 3
C_OUT = 3
H_OUT = 20
W_OUT = 20
TOTAL_OPS = (CI * H_OUT * W_OUT * KERNEL * KERNEL * C_OUT) * 2

def npi_to_json(logFile):
	data = []
	with open(logFile) as f:
		for line in f:
			data.append(json.loads(line))

	result = []
	for ii in range(len(data)):
		record = {'id' : ii}
		time = data[ii]['result'][0][0] * 1e3
		record.update({'time': time})
		record.update({'flop': (TOTAL_OPS/float(time))*1000.0})

		if time > 1e8:
			record.update({'result' : False})
		else:
			record.update({'result' : True})

		result.append(record)
	return result

def azure_to_json(logFile):
	runtime = []
	lineCount = 0
	with open(logFile, 'rb') as f:
		line = "1"
		task_counter = 0
		while(line):
			line = f.readline()
			lineCount += 1
			# print(line)
			line_str = line.decode("utf-8", errors='ignore')
			
			itemDone = False
			if "START" in line_str and len(line) == 9:
				id0 = struct.unpack('B', line[0:1])[0]
				id1 = struct.unpack('B', line[1:2])[0]
				# id = id0*(2**8) + id1
				id = task_counter
				item = {"id" : id}
				itemDone = False
			elif "RES" in line_str and len(line) == 9:
				res = chr(line[7])
				if res == '0':
					result = False
				elif res == '1':
					result = True
				else:
					raise
				# print("res: " + str(result))
				item.update({"result" : result})
				itemDone = False
			elif "TIME" in line_str and len(line) == 19:
				time = line[8:8+10]
				time = time.decode("utf-8")
				time = time.strip('\x00')
				time = float(time)
				item.update({"time" : time})
				item.update({'flop': (TOTAL_OPS/float(time))*1000.0})
				itemDone = True

			if itemDone:
				runtime.append(item)
				task_counter += 1

		print("azure line read: " + str(lineCount))
		print("azure num of tasks: " + str(len(runtime)))
	return runtime

def plot_flops(azure, npi):
	scale = 1e6 * 1.0
	azure_flops = []
	npi_flops = []
	maxVal = 0
	if npi:
		for item in azure:
			npi_item = list(filter(lambda task: task['id'] == item['id'], npi))[0]
			if npi_item['result'] and item['result']:
				f0 = item['flop']
				f1 = npi_item['flop']
				azure_flops.append(f0)
				npi_flops.append(f1)
	else:
		for item in azure:
			if item['result']:
				f0 = item['flop']
				azure_flops.append(f0)

	azure_acc = []
	azure_best = azure_flops[0]
	for item in azure_flops:
		if item >= azure_best:
			azure_acc.append(item)
			azure_best = item
		else:
			azure_acc.append(azure_best)
	azure_plot = [x / scale for x in azure_acc]

	npi_acc = []
	if npi_flops:
		npi_best = npi_flops[0]
		for item in npi_flops:
			if item >= npi_best:
				npi_acc.append(item)
				npi_best = item
			else:
				npi_acc.append(npi_best)
	
	npi_plot = []
	if npi_acc:
		npi_plot = [x / scale for x in npi_acc]
	
	if npi_plot:
		plt.plot(azure_plot, 'r--', npi_plot, 'b--')
		plt.legend(['Azure', 'NPI'])
	else:
		plt.plot(azure_plot, 'r--')
		# plt.legend(['Azure'])
	
	plt.xlim(left=-5)
	plt.ylim(bottom=0, top=max(azure_plot)+100)
	plt.ylabel('MFLOPS')
	plt.xlabel('Trials')
	
	plt.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.99)
	plt.show()

def plot_time(azure, npi):
	time_azure = []
	time_npi = []
	maxVal = 0
	if npi:
		for item in azure:
			npi_item = list(filter(lambda task: task['id'] == item['id'], npi))[0]
			if npi_item['result'] and item['result']:
				t0 = item['time']
				t1 = npi_item['time']
				time_azure.append(t0)
				time_npi.append(t1)
	else:
		for item in azure:
			if item['result']:
				t0 = item['time']
				time_azure.append(t0)
	# azure_norm = [float(i)/max(time_azure) for i in time_azure]
	# npi_norm = [float(i)/max(time_npi) for i in time_npi]
	azure_final = time_azure
	npi_final = time_npi

	print("size of plot: " + str(len(time_azure)))
	print("Azure min task number: " + str(time_azure.index(min(time_azure))) + " and time: " + str(min(time_azure)))
	if time_npi:
		print("NPI min task number: " + str(time_npi.index(min(time_npi))) + " and time: " + str(min(time_npi)))

	plt.subplot(2, 1, 1)
	if npi_final:
		plt.plot(azure_final, 'r*', npi_final, 'bs')
		plt.legend(['Azure', 'NPI'])
	else:
		plt.plot(azure_final, 'r*')
		plt.legend(['Azure'])

	# plt.yscale('log')
	plt.xlim(left=-1)
	plt.ylim(bottom=0)
	plt.ylabel('Time (ms) (log scale)')
	plt.xlabel('Trials')
	plt.legend(['Azure', 'NPI'])
	plt.title("Runtime")

	if npi_final:
		plt.subplot(2, 1, 2)
		plt.scatter(x=azure_final, y=npi_final)
		plt.xlabel('Time Azure (ms)')
		plt.ylabel('Time NPI (ms)')
		plt.xlim(left=0)
		plt.ylim(bottom=0)

	plt.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.99)
	plt.show()

def plot_footprint(opts, build_path=None, runtime_path=None):
	csv_generate = opts.csv

	if not build_path:
		raise ValueError('Build path is not valid!')

	tasks_dirs = [dI for dI in os.listdir(build_path) if os.path.isdir(os.path.join(build_path, dI))]
	tasks_dirs.sort()
	num_of_tasks = len(tasks_dirs)
	all_tasks_size = []
	for ii in range(len(tasks_dirs)):
		task_path = os.path.join(build_path, tasks_dirs[ii])
		
		schedule_dirs = [dI for dI in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, dI))]
		schedule_dirs.sort()
		
		task_size = []
		for jj in range(len(schedule_dirs)):
			schedule_path = os.path.join(task_path, schedule_dirs[jj])
			output_path = os.path.join(schedule_path, 'build')
			
			elf_file = os.path.join(output_path, 'octoml_AS.out')
			params_file = os.path.join(output_path, 'conv2d_params.bin')
			graph_file = os.path.join(output_path, 'conv2d_graph.bin')

			elf_size = os.path.getsize(elf_file)
			params_size = os.path.getsize(params_file)
			graph_size = os.path.getsize(graph_file)
			total = sum([elf_size, params_size, graph_size])
			# total = sum([params_size])
			# print(elf_size, params_size, graph_size)
			# print([elf_size, params_size, graph_size])
			# print(total)
			task_size.append(total)

		all_tasks_size.append(task_size)
	
	#create CSV of time and footprint
	if runtime_path:
		if not runtime_path:
			raise ValueError('Runtimepath is not valid!')

		for ii in range(num_of_tasks):
			azure_results = azure_to_json(os.path.join(runtime_path, f'task_{str(ii)}.dump'))

			time_footprint = []
			task_footprint = all_tasks_size[ii]
			for jj in range(len(azure_results)):
				item = azure_results[jj]
				time = item['time']
				size = task_footprint[jj] / 1024	#convert to KB
				time_footprint.append([time, size])

			if csv_generate:
				csv_path = os.path.join(build_path, f'fooprint_task_{str(ii).zfill(4)}.csv')
				with open(csv_path, 'w') as csv_file:
					wr = csv.writer(csv_file, delimiter=',')
					for item in time_footprint:
						wr.writerow(item)
					print(f'File {csv_path} generated!')
	
	return all_tasks_size

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file', default=None)
	parser.add_argument('-s', '--source', default=None)
	parser.add_argument('--footprint', action='store_true')
	parser.add_argument('--csv', action='store_true')
	opts = parser.parse_args()

	azure = []
	npi = []

	if opts.file:
		azure = azure_to_json(logFile=opts.file)

	if opts.source:
		npi = npi_to_json(logFile=opts.source)

	if opts.source or opts.file:
		plot_time(azure=azure, npi=npi)
		plot_flops(azure=azure, npi=npi)
	
	if opts.footprint:
		# items = plot_footprint(build_path='build',
		# 					   runtime_path='/home/parallels/azure-sphere/tuning/server/cifar'
		# 					   )
		items = plot_footprint(opts=opts, build_path='build',
							   runtime_path='/home/parallels/azure-sphere/tuning/server/keyword')
		for ii in range(len(items)):
			task = items[ii]
			print(task)
			# print("\n")
			sch_min = min(task)
			sch_min_ind = task.index(sch_min)
			print(f'task: {ii},\tmin: {sch_min},\tindex: {sch_min_ind}')
			
		

	# count = 0
	# for item in npi:
	# 	if item["result"] == False:
	# 		count += 1
	# print(count) 

	# test = []
	# for item in azure:
	# 	if item['result']:
	# 		test.append(item["time"])
	# print(len(test))
	# print(statistics.mean(test))
	# print(statistics.stdev(test))
