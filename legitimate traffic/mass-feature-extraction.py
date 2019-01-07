import sys	
import subprocess
import gzip
import time
import json

if len(sys.argv) < 2:
	print("Usage: python mass-feature-extraction.py <start directory>")
	print("Expected: start directory contains folders, within each contains folders, within each contains pcap files.")

directory = sys.argv[1]

subdirectories = subprocess.check_output(["ls", directory]).decode("utf-8").split("\n");

for i in range(0, len(subdirectories) - 1):
	subsubdirectories = subprocess.check_output(["ls", directory + "/" + subdirectories[i]]).decode("utf-8").split("\n")

	for j in range(0, len(subsubdirectories) - 1):
		fileNames = subprocess.check_output(["ls", directory + "/" + subdirectories[i] + "/" + subsubdirectories[j]]).decode("utf-8").split("\n")

		for k in range(0, len(fileNames) - 1):
			fileName = directory + "/" + subdirectories[i] + "/" + subsubdirectories[j] + "/" + fileNames[k];
			print("Processing file: " + fileName);

			subprocess.check_output(["./joy", "bidir=1", "http=1", "tls=1", "dns=1", "ppi=1", "output=output.gz", fileName]);
			
			f = gzip.open('output.gz', 'rb')
			file_content = f.read()
			f.close()

			jsonData = file_content.split("\n")
			#Find the one that is correct
			connections = []
			for l in range(1, len(jsonData) - 1):
				if "tls" in json.loads(jsonData[l]):
					connections.append(l)

			data = {}
			if len(connections) == 0:
				for l in range(1, len(jsonData) - 1):
					connections.append(l)

			if len(connections) == 1:
				data = json.loads(jsonData[connections[0]])
			else:
				print("Warning: Length of connections != 0. It is: " + str(connections))
				longestInt = connections[0]
				longest = len(jsonData[connections[0]])
				for l in range(1, len(connections)):
					if longest < len(jsonData[connections[l]]):
						longestInt = connections[l]
						longest = len(jsonData[connections[l]])

				print("Guessing: " + str(longestInt))
				data = json.loads(jsonData[longestInt])

			packets = []

			packetInformation = data["ppi"]
			packetProtocol = [0, 0, 0, 0, 0, 0] #TCP, SSL2, SSL3, TLS1, TLS2, TLS3
			#Hacky fixes
			if "tls" in data:
				if "s_version" in data["tls"]:
					protocol = data["tls"]["s_version"] # unknown = 0, SSLv2 = 1, SSLv3 = 2, TLS1.0 = 3, TLS1.1 = 4, TLS1.2 = 5."
					if protocol != 0:
						packetProtocol[protocol] = 1

			previous = 0
			for l in range(0, len(packetInformation)):
				singlePacket = {}
				singlePacket["Come"] = 0
				if packetInformation[l]["dir"] == ">":
					singlePacket["Come"] = 1

				singlePacket["Protocol"] = [0, 0, 0, 0, 0, 0] #TCP, SSL2, SSL3, TLS1, TLS2, TLS3
				singlePacket["Length"] = packetInformation[l]["b"]
				singlePacket["Interval"] = packetInformation[l]["t"] - previous
				singlePacket["Windows Size"] = 0

				flagList = [0, 0, 0, 0, 0, 0, 0, 0, 0] #Enum: NS, CWR, ECE, URG, ACK, PSH, RST, SYN, FIN
				#Source: https://github.com/cisco/joy/blob/master/src/ppi.c
				packetFlags = packetInformation[l]["flags"]
				if "C" in packetFlags:
					flagList[1] = 1
				if "E" in packetFlags:
					flagList[2] = 1
				if "U" in packetFlags:
					flagList[3] = 1
				if "A" in packetFlags:
					flagList[4] = 1
				if "P" in packetFlags:
					flagList[5] = 1
					singlePacket["Protocol"] = packetProtocol
				else:
					singlePacket["Protocol"][0] = 1
				if "R" in packetFlags:
					flagList[6] = 1
				if "S" in packetFlags:
					flagList[7] = 1
				if "F" in packetFlags:
					flagList[8] = 1
				if not all(c in ('C', 'E', 'U', 'A', 'P', 'R', 'S', 'F') for c in packetFlags): 
					print('Unrecognized character in packet: ' + packetFlags)

				singlePacket["Flag"] = flagList

				packets.append(singlePacket);
				previous = packetInformation[l]["t"]

			with open("features.csv", 'a+') as f_out:
				for l in range(0, len(packets)):
					f_out.write("[" + str(packets[l]["Protocol"])[1:-1] + ", " + str(packets[l]["Length"]) + ", " + str(packets[l]["Interval"]) + ", " + str(packets[l]["Windows Size"]) + ", " + str(packets[l]["Flag"])[1:-1] + ", " + str(packets[l]["Come"]) + "], ")
				f_out.write("\r\n")