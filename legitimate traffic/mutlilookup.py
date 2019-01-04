#!/usr/bin/python

import sys
from scapy.all import *
from threading import Thread, Event
from time import sleep
import traceback
import xlrd
import socket
import requests
import datetime
import os
import logging
from fake_useragent import UserAgent
#from prob_matrix import main

logging.basicConfig(filename='application.log',level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')

if len(sys.argv) <= 1:
	print("Usage: <file.xlsx>")
	exit(1)

file_location = sys.argv[1]
workbook = xlrd.open_workbook(file_location)
sheet = workbook.sheet_by_index(0)
websites = []
names = []

for row in range(sheet.nrows):
	if(sheet.cell_value(row, 2) is not ""):
		ipList = []
		ipList.append(sheet.cell_value(row, 2))
		thisWebsite = sheet.cell_value(row, 2)

		if thisWebsite.find("https") != -1:
			thisWebsite = thisWebsite[8:]
		else:
			if thisWebsite.find("http") != -1:
				thisWebsite = thisWebsite[7:]
		if thisWebsite.find("/") != -1:
			domain, extra = thisWebsite.split('/', 1)
		else:
			domain = thisWebsite

		try:
			addrList = socket.getaddrinfo(domain, None)
		except socket.gaierror as e:
			with open("errors.out", 'a+') as f:
				f.write("===========================================================================================================================\n");
				f.write(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + " - Error accessing website:\n")
				f.write(traceback.format_exc());
				f.write("Website: " + str(domain) + "\n");
			logging.warning("DNS error for website: " + str(domain));
			addrList = []

		names.append(sheet.cell_value(row, 0))
		ipList.append(domain)
		for item in addrList:
			ipList.append(item[4][0])
		websites.append(ipList)

with open("ip-domain.csv", 'a+') as f:
	for i in range(0, len(websites)):
		f.write(websites[i][0] + "," + websites[i][1] + "," + websites[i][2] + "\n");