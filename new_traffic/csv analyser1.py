import csv


col_header1= [
    "","heartbleed",
              "CCS",
              "ticketbleed",
              "ROBOT",
              "secure_renego",
              "secure_client_renego",
              "CRIME_TLS",
              "BREACH",
              "POODLE_SSL",
              "fallback_SCSV",
              "SWEET32",
              "FREAK",
              "DROWN",
              "LOGJAM",
              "LOGJAM-common_primes",
              "BEAST_CBC_TLS1",
              "BEAST",
              "LUCKY13",
              "RC4"
]

with open("23k.csv","r") as g:
    reader= csv.reader(g,dialect="excel")
    for i in reader :
        col= len(col_header1)
        print(col)
        for g in range(0,col):
            h= (g*5)-1
            if i[h]=="1":
                print(h)
                print(i)
                filename=col_header1[g]+".csv"
                print(filename)
                with open(filename, "a", newline="") as c:
                    w=csv.writer(c,dialect="excel")
                    w.writerow([i[0]])
