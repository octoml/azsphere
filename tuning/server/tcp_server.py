import socket
import sys
import atexit
import logging
import json
from enum import Enum
import argparse

IP = '192.168.0.10'
PORT = 11000

class NetworkState(Enum):
	WAITING = 1
	STARTED = 2
	RESULT = 3
	TIME = 4

def exitHandler():
	print("closing socket!")
	sock.close()

def server_start(opts):
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
	        while True:
	            data = connection.recv(100)
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
	opts = parser.parse_args()

	server_start(opts)