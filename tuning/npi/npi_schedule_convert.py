import json
import os
import shutil
import argparse

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
		filename = "schedule_" + str(count).zfill(4) + ".txt"
		count += 1
		filepath = os.path.join(path, filename)
		with open(filepath, 'w+') as f:
			tmp = json.dumps(log)
			# print(type(tmp))
			# print(tmp)
			f.write(tmp)
		f.close()



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--source', default='')
	parser.add_argument('-d', '--destination', default='')
	opts = parser.parse_args()

	jsonData = get_log(filename=opts.source)
	log_generate(jsonData, path=opts.destination)