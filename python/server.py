import socket
import sys

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
server_address = ('192.168.0.10', 11000)
# server_address = ('10.42.0.1', 10000)
print('starting up on ', server_address[0], ' port ', server_address[1])
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

while True:
    # Wait for a connection
    print('waiting for a connection')
    connection, client_address = sock.accept()

    try:
        print('connection from', str(client_address))

        # Receive the data in small chunks and retransmit it
        while True:
            data = connection.recv(16)
            print('received ', data)
            if data:
                continue
                # print('sending data back to the client')
                # connection.sendall(data)
            else:
                print('no more data from ', client_address)
                break
            
    finally:
        # Clean up the connection
        connection.close()