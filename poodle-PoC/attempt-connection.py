#!/usr/bin/python
# -*- coding: utf-8 -*- 

import socket
import OpenSSL
from OpenSSL import *
import sys

serverName = sys.argv[1]
print("Using server : " + serverName + ", port " +  sys.argv[2])

#context = OpenSSL.SSL.Context(SSL.SSLv3_METHOD)
#soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#soc.settimeout(CONNECTION_TIMEOUT)
#connection = OpenSSL.SSL.Connection(context,soc)
#connection.connect((host,port))
#connection.do_handshake()

ctx = SSL.Context(SSL.SSLv3_METHOD)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((serverName, sys.argv[2]))
sslSocket = socket.ssl(s)

print ("writing socket..")
sslSocket.write('<transaction><data>14</data></transaction>\n')
s.close()