import socket
import sys
import atexit
import logging
import json
from enum import Enum
import argparse
import os

IP = '192.168.0.10'
PORT = 11000
sock = None
REC_BUFF_SIZE = 2000

class NetworkState(Enum):
	WAITING = 1
	STARTED = 2
	RESULT = 3
	TIME = 4

def exitHandler():
	global sock
	print("closing socket!")
	sock.close()

def server_start(opts):
	global sock
	# Create a TCP/IP socket
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

	# Bind the socket to the port
	server_address = (IP, PORT)
	print('starting up on ', server_address[0], ' port ', server_address[1])
	sock.bind(server_address)

	# Listen for incoming connections
	sock.listen(1)

	atexit.register(exitHandler)
	log_file = open(opts.log, "wb+", buffering=0)

	numOfConnection = 1

	if opts.demo:
		#get demo files and data
		wav_files = []
		for file in os.listdir(opts.wav):
			if file.endswith('.wav'):
				wav_files.append(file)
		# wav_files = ['2_yes.wav']
		print(wav_files)

		wav_data = [None] * len(wav_files)
		for ii, item in enumerate(wav_files):
			wav_data[ii] = prepare_input(os.path.join(opts.wav, item)).astype(np.float32).tobytes()

	while True:
		# Wait for a connection
		print('waiting for a connection')
		connection, client_address = sock.accept()

		try:
			print('connection from', str(client_address))
			state = NetworkState.WAITING
			print("Connection: " + str(numOfConnection))
			numOfConnection += 1
			# Receive the data in small chunks and retransmit it

			if opts.demo:
				data_counter = 0
				while True:
					data = connection.recv(REC_BUFF_SIZE)
					if len(data) > 0:
						print("data: " + str(data))
					if data:
						if 'ready' in str(data):
							connection.send(wav_data[data_counter])
							data_counter += 1
							data_counter = data_counter % len(wav_data)
						else:
							continue
					else:
						print('no more data from ', client_address)
						break
			else:
				while True:
					data = connection.recv(REC_BUFF_SIZE)
					print("data: " + str(data))
					if data:
						log_file.write(data)
					else:
						print('no more data from ', client_address)
						break
		finally:
			# Clean up the connection
			connection.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--log', default='')
	parser.add_argument('--demo', action='store_true')
	parser.add_argument('--wav', default='', help='Path to a directory with WAV files')
	opts = parser.parse_args()

	if opts.demo:
		from model.keyword_spotting import prepare_input
		import numpy as np
	
	server_start(opts)