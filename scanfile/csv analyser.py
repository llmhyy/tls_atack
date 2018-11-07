import csv
data=[]
row = 1
ip=""
index =0
write_data=[]
col_header1= ["heartbleed",
              "CCS",
              "ticketbleed",
              "ROBOT",
              "secure_renego",
              "secure_client_renego",
              "CRIME_TLS,BREACH,POODLE_SSL",
              "fallback_SCSV",
              "SWEET32",
              "FREAK",
              "DROWN",
              "LOGJAM",
              "LOGJAM-common_primes",
              "BEAST_CBC_TLS1",
              "BEAST","LUCKY13",
              "RC4"
]
col_header2 =["ok","LOW", "MEDIUM","HIGH","Comment"]
lin1 = ["IP"]
lin2=["IP"]
for l in col_header1:
    for i in range(0,6):
        lin1.append(l)
    for g in col_header2:
        lin2.append(g)

print(lin1)
print(lin2)
with open("report/Food.csv", "w") as report:
    reader=csv.writer(report, dialect="excel")
    reader.writerow(lin1)
    reader.writerow(lin2)
def info_b(info):
    r_msg= [0,0,0,0]
    if info == "OK":
        r_msg = [1,0,0,0]#[OK, Low, Mediul
    elif info =="LOW":
        r_msg =[0,1,0,0]
    elif info =="MEDIUM":
        r_msg =[0,0,1,0]
    elif info=="HIGH":
        r_msg =[0,0,0,1]
    else:
        r_msg =None

    return r_msg
with open("data/Food_r.csv", "r") as ip1:
    reader = csv.reader(ip1)
    for i in reader:
        data.append(i)
something = []
for i in range(len(data)):
    row_data = data[i]
    print("i = %d",format(i))
    if i!=1:
        
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

            if info!=None:
                for x in info:
                    write_data.append(x)
                write_data.append(row_data[4])

            print("write_data" + str(len(write_data)))
            if i == len(data)-1:
                with open("report/Food.csv", "a") as report:
                    writer=csv.writer(report, dialect="excel")
                    writer.writerow(write_data)
   
    
        elif(row_data[0]=="service"):
            with open("report/Food.csv", "a") as report:
                writer=csv.writer(report, dialect="excel")
                writer.writerow(write_data)
            del write_data[:]
            print("initial")
            print(write_data)
