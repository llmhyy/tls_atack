import subprocess
import csv
import os
ip =""
with open("data/Finance.csv","r") as sites:
    reader = csv.reader(sites)
    for i in reader:
        ip = i[1]
        print(ip)
        st ="./testssl.sh --csvfile data/Finance_r.csv --append -U " + ip
        print(st)
        os.system (st)
        #scan = subprocess.run(["./testssl.sh",
         #                      "csvfile" ,"ip1.csv" ,
          #                     "--append"+"-U " + ip],
           #                                   stdout=subprocess.PIPE,
            #                                  stderr = subprocess.PIPE,
             #                                 timeout =20)
        #print(scan.stdout)
    

