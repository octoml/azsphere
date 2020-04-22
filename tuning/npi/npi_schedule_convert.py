import json
import os
import shutil

def get_log(filename):
	data = []
	with open(filename) as f:
		for line in f:
			data.append(json.loads(line))
	return data

def log_generate(jsonData, path):
	if os.path.exists(path):
		shutil.rmtree(path)

	if not os.path.exists(path):
		os.mkdir(path)

	count = 0
	for log in jsonData:
		filename = "hist_log_" + str(count).zfill(4) + ".log"
		count += 1
		filepath = os.path.join(path, filename)
		with open(filepath, 'w+') as f:
			tmp = json.dumps(log)
			# print(type(tmp))
			# print(tmp)
			f.write(tmp)
		f.close()



if __name__ == '__main__':
	jsonData = get_log("benchmarks/npi.conv2d.log.tmp")
	# print(len(jsonData))
	# print(jsonData[0])
	log_generate(jsonData, path="npi400")