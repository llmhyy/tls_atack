
from os import listdir
from os.path import isfile,join
from shutil import move
import os

onlyfiles= [f for f in listdir("sum/") if isfile(join('sum/',f))]
for i in range(len(onlyfiles)):
    src= "sum/"+onlyfiles[i]
    dis= str(i//100+1)+'/'
    if not os.path.exists(dis):
        os.makedirs(dis)
        move(src,dis)
    else:
        move(src,dis)

