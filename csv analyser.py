import csv
data=[]
row = 1
ip=""
index =0
write_data=[]
def info_b(info):
    r_msg= [0,0,0,0]
    if info == "OK":
        r_msg = [1,0,0,0]#[OK, Low, Medium, High]
        
    elif info =="LOW":
        r_msg =[0,1,0,0]
    elif info =="MEDIUM":
        r_msg =[0,0,1,0]
    elif info=="HIGH":
        r_msg =[0,0,0,1]
    else:
        r_msg =[0,0,0,0]

    return r_msg
with open("ip1.csv", "r") as ip1:
    reader = csv.reader(ip1)
    for i in reader:
        data.append(i)
something = []
for i in range(len(data)):
    row_data = data[i]
    if i!=0:
        
        something = row_data
        print(str(i)+ str(len(row_data))+"\n")
        print(row_data[4])
        print(type(row_data[4]))
        if (row_data[0]!="service"):
            if ip!=row_data[1]:
                ip= row_data[1]
                write_data= [ip] + write_data
                print(write_data)
        
        info= info_b(row_data[3])
        for x in info:
            write_data.append(x)
        write_data.append(row_data[4])

        print("write_data" + str(len(write_data)))
   
    
        if (row_data[0]=="service"):
            with open("report.csv", "a") as report:
                writer=csv.writer(report, dialect="excel")
                writer.writerow(write_data)
            del write_data[:]                     
