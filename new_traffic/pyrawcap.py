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

if len(sys.argv) <= 3:
	print("Usage: <file.xlsx> <repetitions> <sniff-output-prefix>")
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

class Sniffer(Thread):
	def  __init__(self, host, interface=None):
		super().__init__()

		self.daemon = True

		self.socket = None

		self.filters = "host " + str(host)

		self.interface = interface
		self.stop_sniffer = Event()

	def run(self):
		self.socket = conf.L2listen(
			type=ETH_P_ALL,
			iface=self.interface,
			filter=self.filters
		)

		self.capture = sniff(
			opened_socket=self.socket,
			prn=self.print_packet,
			stop_filter=self.should_stop_sniffer#,
			#store=0
		)

	def join(self, timeout=None):
		self.stop_sniffer.set()
		super().join(timeout)

	def should_stop_sniffer(self, packet):
		return self.stop_sniffer.isSet()

	def print_packet(self, packet):
		ip_layer = packet.getlayer(IP)
		logging.debug("[!] New Packet: {src} -> {dst}".format(src=ip_layer.src, dst=ip_layer.dst))

i = 0
retries = 0
numOfTries = int(sys.argv[2])
outputPrefix = str(sys.argv[3])
failures = 0
httpFailures = 0
ua = UserAgent()
failedNames = []
failedWebsites = []
while i < len(websites):
	sniffer = Sniffer(websites[i][1])
	logging.info("Now processing: " + websites[i][0])

	logging.debug("[*] Start sniffing...")
	sniffer.start()

	j = 0
	httpRetries = 0
	timeouts = 0
	try:
		while j < numOfTries:
			try:
				headers = {'User-Agent': ua.random}
				r = requests.get(websites[i][0], timeout=30, headers=headers);
			except requests.Timeout:
				timeouts = timeouts + 1;
				logging.info("Timeout #" + str(timeouts) + " for website: " + str(websites[i][0]));
				if timeouts < 3:
					continue
				else:
					logging.warning("Failed 6 timeouts for website: " + str(websites[i][0]));
					raise TimeoutError("Timeout for website: " + str(websites[i][0]));

			if r.status_code == 200:
				j += 1
				logging.debug(j)
			else:
				if httpRetries < 10:
					httpRetries += 1
					logging.info("Failed request for: " + str(websites[i][0]) + ", status code: " + str(r.status_code) + ", retry number: " +  str(httpRetries))
				else:
					logging.warning("Failed " + str(httpRetries) + " requests for: " + str(websites[i][0]) + ", status code: " + str(r.status_code) + ", skipping...")
					httpFailures += 1
					failedNames.append(names[i])
					failedWebsites.append(websites[i][0])
					break;
	except Exception as e: 
		with open("errors.out", 'a+') as f:
			f.write("===========================================================================================================================\n");
			f.write(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + " - Error accessing website:\n")
			f.write(traceback.format_exc());
			f.write("Website: " + str(websites[i][0]) + "\n");
		logging.warning("Request error for website: " + str(websites[i][0]));
		failedNames.append(names[i])
		failedWebsites.append(websites[i][0])
		i += 1
		continue

	logging.debug("[*] Stop sniffing")
	sniffer.join(5.0)

	if not os.path.exists(outputPrefix):
		os.makedirs(outputPrefix)

	if hasattr(sniffer, 'capture'):
		filename = outputPrefix + names[i] + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pcap"
		if j < numOfTries:
			filename = outputPrefix + names[i] + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(j) + ".pcap"

		wrpcap(filename, sniffer.capture)

		if sniffer.isAlive(): #TODO fix: Note that if it enters the if statement then it will crash in the next step when it attempts to get sniffer.capture.
			sniffer.socket.close()

		i += 1
		retries = 0

		#try:
		#	main(filename, filename)
		#except Exception as e:
		#	with open("errors.out", 'a+') as f:
		#		f.write("===========================================================================================================================\n");
		#		f.write(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + " - Error running prob_matrix:\n")
		#		f.write(traceback.format_exc());
		#		f.write("Website: " + str(websites[i][0]) + "\n");
		#		logging.warning("Request error for prob_matrix: " + str(websites[i][0]));
		#		failedNames.append(names[i])
		#		failedWebsites.append(websites[i][0])
		#		i = i + 1
		#	continue

	else:
		if retries < 0:
			logging.info("Sniffer does not have capture property, retrying...")
			retries += 1
		else:
			logging.warning("Failure: Maximum retries exceeded for website " + str(websites[i][0]) + " Skipping...")
			i += 1
			failures += 1
			failedNames.append(names[i])
			failedWebsites.append(websites[i][0])

with open(file_location[:-5] + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv", 'a+') as f:
	for i in range(0, len(failedNames)):
		f.write(failedNames[i] + ",," + failedWebsites[i] + "\n");

logging.info("py raw cap complete. " + str(failures) + " out of " + str(len(websites)) + " failed")
