import csv

data=[]
data2=[]
with open("9k.csv","r") as f:
    reader = csv.reader(f, dialect="excel")
    for i in reader:
        if i[0] not in data:

            data.append(i[0])
with open("23k.csv","r") as r:
    reader = csv.reader(r, dialect="excel")
    for g in reader:
 
        if g[0] not in data:
            data2.append(g[0])
