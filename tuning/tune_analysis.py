import struct
import argparse
import json
import matplotlib.pyplot as plt

def npi_to_json(logFile):
	data = []
	with open(logFile) as f:
		for line in f:
			data.append(json.loads(line))

	result = []
	for ii in range(len(data)):
		record = {'id' : ii}
		time = data[ii]['result'][0][0]
		record.update({'time': time})

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
		while(line):
			line = f.readline()
			lineCount += 1
			# print(line)
			line_str = line.decode("utf-8", errors='ignore')
			
			itemDone = False
			if "START" in line_str and len(line) == 9:
				id0 = struct.unpack('B', line[0:1])[0]
				id1 = struct.unpack('B', line[1:2])[0]
				id = id0*(2**8) + id1
				# print('id = ' + str(id0) + ' ' + str(id1))
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
			elif "TIME" in line_str and len(line) == 13:
				time0 = struct.unpack('B', line[8:9])[0] 	* (2**24)
				time1 = struct.unpack('B', line[9:10])[0] 	* (2**16)
				time2 = struct.unpack('B', line[10:11])[0] 	* (2**8)
				time3 = struct.unpack('B', line[11:12])[0]

				time = time0 + time1 + time2 + time3
				item.update({"time" : time})
				itemDone = True

			if itemDone:
				runtime.append(item)

		print("azure line read: " + str(lineCount))
		print("azure num of tasks: " + str(len(runtime)))
	return runtime

def plot_time(azure, npi):
	time_azure = []
	time_npi = []
	maxVal = 0
	for item in azure:
		npi_item = list(filter(lambda task: task['id'] == item['id'], npi))[0]
		if npi_item['result'] and item['result']:
			t0 = item['time']
			t1 = npi_item['time']
			time_azure.append(t0)
			time_npi.append(t1)

	azure_norm = [float(i)/max(time_azure) for i in time_azure]
	npi_norm = [float(i)/max(time_npi) for i in time_npi]
	print(len(time_azure))
	plt.scatter(azure_norm, npi_norm)
	
	print("Azure: " + str(azure_norm.index(min(azure_norm))))
	print("NPI: " + str(npi_norm.index(min(npi_norm))))

	# plt.plot(azure_norm, 'r*', npi_norm, 'bs')
	# plt.ylabel('Azure')
	# plt.xlabel('Npi')
	plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file', default='')
	parser.add_argument('-s', '--source', default='')
	opts = parser.parse_args()

	azure = azure_to_json(logFile=opts.file)
	npi = npi_to_json(logFile=opts.source)

	plot_time(azure=azure, npi=npi)
	# npi_time = []
	# for item in npi:
	# 	npi_time.append(item['time'])

	# plt.plot(npi_time)
	# plt.ylabel('Time (ms)')
	# plt.show()
	# for item in runtime:
	# 	print(item)