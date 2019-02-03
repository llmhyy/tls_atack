import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Input directory containng file parts to be joined', required=True)
args = parser.parse_args()

full_path = args.input
path, foldername = os.path.split(os.path.normpath(full_path))
new_full_path = os.path.join(path, foldername+'.csv')

with open(new_full_path, 'w') as out_f:
	for filename in sorted(os.listdir(full_path)):
		with open(os.path.join(full_path,filename), 'r') as in_f:
			for line in in_f:
				out_f.write(line)
