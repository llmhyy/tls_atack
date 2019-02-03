import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Input file to be split', required=True)
args = parser.parse_args()

path, filename = os.path.split(args.input)
real_filename, fileextension = os.path.splitext(filename)
os.mkdir(os.path.join(path, real_filename))

with open(filename,'r') as f:
    count = 1
    new_filename = os.path.join(path, real_filename, real_filename+'_'+str(count)+fileextension)
    out_f = open(new_filename, 'w')

    for line in f:
        if os.path.getsize(new_filename)>99000000:
            out_f.close()
            count+=1
            new_filename = os.path.join(path, real_filename, real_filename+'_'+str(count)+fileextension)
            out_f = open(new_filename,'w')
        else:
            out_f.write(line)
    out_f.close()




