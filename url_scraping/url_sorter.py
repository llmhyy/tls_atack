import csv

domain = []

with open("urls.csv", "r") as f:
    reader= csv.reader(f, dialect= "excel")
    for i in reader:
        f_string= i[0]
        c_string= f_string[8:-1]
        print(len(i))
        index=c_string.find("/")
        c_string=c_string[0:index]
        if c_string not in domain:
            domain.append(c_string)
            with open("refined.csv", "a", newline='') as g :
                h = csv.writer(g, dialect="excel")
                h.writerow(i)
        print(type(i))
        print(c_string)
   
        
