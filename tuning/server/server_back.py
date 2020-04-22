import socket
import sys
import atexit
import logging
import json
from enum import Enum


class NetworkState(Enum):
	WAITING = 1
	STARTED = 2
	RESULT = 3
	TIME = 4

def exitHandler():
	print("closing socket!")
	sock.close()


IP = '192.168.0.10'
PORT = 11000

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Bind the socket to the port
server_address = (IP, PORT)
print('starting up on ', server_address[0], ' port ', server_address[1])
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

atexit.register(exitHandler)
# logging.basicConfig(filename='conv2d_as.log', level=logging.INFO)
# logging.info('Started')
data_log = open("conv2d_as.log", "a+", buffering=1)
# data_log.write("Hello")

while True:
    # Wait for a connection
    print('waiting for a connection')
    connection, client_address = sock.accept()

    try:
        print('connection from', str(client_address))
        res = {}
        state = NetworkState.WAITING
        # Receive the data in small chunks and retransmit it
        while True:
            data = connection.recv(100)
            print("data: " + str(data))
            
            try:
            	data_str = data.decode("utf-8")
            # data_str = unicode(data, errors='ignore')
            except:
            	print(data)
            	data_str = data.decode("utf-8", errors='ignore')

            print(data_str)

            if data_str:
            	dataList = data_str.split(',')
            	if len(dataList) >= 2:
	            	if dataList[1] == "START":
	            		res.update({"id": dataList[0]})
	            		state = NetworkState.STARTED
	            	elif dataList[1] == "RES":
	            		if dataList[2] == "0":
	            			res.update({"result" : False})
	            		elif dataList[2] == "1":
	            			res.update({"result" : True})
	            		else:
	            			print("ERRRORRRRRRRRR")
	            	elif dataList[1] == "TIME":
	            		if state == NetworkState.STARTED:
	            			time = data[9]*(2^24) + data[10]*(2^16) + data[11]*(2^8) + data[12]
	            			state = NetworkState.TIME
	            			print(time)
	            			res.update({"time" : time})
	            			break
	            	else:
	            		print(data)
            else:
                print('no more data from ', client_address)
                break
        if state == NetworkState.TIME:
        	data_log.write(json.dumps(res) + "\n")

    finally:
        # Clean up the connection
        print(res)
        connection.close()