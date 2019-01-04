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

			data = json.loads(file_content.split("\n")[1])
			packets = []

			packetInformation = data["ppi"]
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
					singlePacket["Protocol"][0] = 1
				else:
					singlePacket["Protocol"][5] = 1
				if "P" in packetFlags:
					flagList[5] = 1
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

			with open(fileName[:-5] + "-features.json", 'w+') as f_out:
				f_out.write(json.dumps(packets))