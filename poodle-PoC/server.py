#!/usr/bin/env python

import socket
import binascii

TCP_IP = '192.168.1.197'
TCP_PORT = 8080
BUFFER_SIZE = 1024  # Normally 1024, but we want fast response

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)

print("Waiting for connection...");

conn, addr = s.accept()
print('Connection address:' + str(addr))
times = 0
while 1:
	data = conn.recv(BUFFER_SIZE)
	if not data: break
	if times == 0:
		print("received data:" + str(data))
	else:
		print("received data:" + data.decode("utf-8"))
	times = times + 1;
	sendData = "receive"
	conn.send(sendData.encode())  # echo
conn.close()