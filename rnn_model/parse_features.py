import json

def get_features(filename):

	traffics = []

	with open(filename, 'r') as f_in:
		f = f_in.readlines()
		content = [x.strip() for x in f]
		for singleLine in content:
			list2 = []
			packet = singleLine.split("], [")

			for i in range(0, len(packet)):
				data = packet[i]
				if i == 0: #fix for first
					data = data[1:]
				if i == len(packet) - 1:
					data = data[:-2] #fix for last

				packetData = data.split(", ")

				list3 = []
				for j in range(0, len(packetData)):
					list3.append(float(packetData[j]))

				list2.append(list3)

			traffics.append(list2)

	#print(json.dumps(traffics))
	return json.dumps(traffics)