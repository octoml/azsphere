import socket
import sys
from time import sleep

REC_BUFF_SIZE = 2000
# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
# server_address = ('10.42.0.1', 10000)
server_address = ('192.168.0.10', 11000)
# print >>sys.stderr, 'connecting to %s port %s' % server_address
sock.connect(server_address)

# sock.sendall(b"Hello")
sock.sendall(b'ready')
while(True):
	data = repr(sock.recv(REC_BUFF_SIZE))
	
	if len(data) > 0:
		print(data)
		break
	# sleep(10)
	# sock.sendall(b'ready')
	# sleep(10)
	# sock.sendall(b'ready')