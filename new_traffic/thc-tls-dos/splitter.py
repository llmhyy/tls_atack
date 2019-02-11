
from os import listdir
from os.path import isfile,join
from shutil import move
import os
def movefile(files,frm):
    print(len(files))
    for i in range(len(files)):
        src= str(frm)+"/"+files[i]
        print(src)
        dis= 'sum/'
        if not os.path.exists(dis):
            os.makedirs(dis)
            move(src,dis)
        else:
            move(src,dis)
for i in range (1,70):
    onlyfiles= [f for f in listdir(str(i)+"/") if isfile(join(str(i)+'/',f))]
    print(len(onlyfiles))
    movefile(onlyfiles , i)
