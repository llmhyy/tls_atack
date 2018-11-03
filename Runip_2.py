import subprocess
import csv

result= subprocess.run(["./testssl.sh", "-U","157.240.13.35:443"],
                       stdout= subprocess.PIPE,
                       stderr = subprocess.PIPE)
#testing with facebook
y= result.stdout
x= result.stderr
yh=y.decode("utf-8")
poodle_string ="POODLE, SSL\x1b[m (CVE-2014-3566) "
NON_valid = "Unable to open a socket"
data_set=[]
working_set=[]

def scan():
    for ip1 in range(1,9):
        print(ip1)
        for ip2 in range(0,256):
            print(ip2)
            for ip3 in range(0,256):
                print(ip3)
                for ip4 in range(0,256):

                    ip= str(ip1)+"."+str(ip2)+"."+str(ip3)+"."+str(ip4)+":443"
                    print(ip)
                    scan= None
                    try :
                        scan = subprocess.run(["./testssl.sh","-O",ip],
                                          stdout=subprocess.PIPE,
                                          stderr = subprocess.PIPE,
                                          timeout =20)
                    except Exception as a:
                        print(a)
                    if scan is not None:
                        print(scan.stdout)
                        scan_string = scan.stdout.decode("utf-8")

                        if NON_valid in scan_string:
                            print(scan_string)
                            this_ip = {ip :"invalid ip"}
                            print(this_ip)
                        else:
                            Return_Message =MessageParser(IP=ip, Message= scan_string)
                            if Return_Message is not None:
                                        with open("ip1.csv", "a") as f :
                                            writer= csv.writer(f,dialect = "excel")
                                            writer.writerow(Return_Message)
                            else :
                                print ("Nothing")


def MessageParser (IP, Message):
    Test_string= "Testing vulnerabilities"
    F_index= Message.find(Test_string)
    if F_index == -1 :
        return None
    Message= Message[F_index:-1]
    print(Message)

    HB_index=Message.find("Heartbleed[m (CVE-2014-0160)")#HeartBleed
    CSS_index= Message.find("CCS[m (CVE-2014-0224)")#CSS
    HB_string= Message[HB_index:CSS_index]
    TB_index = Message.find("Ticketbleed[m (CVE-2016-9244)")#Tciektbleed
    CSS_string= Message[CSS_index: TB_index]
    SR_index = Message.find("Secure Renegotiation [m(CVE-2009-3555)")#Secure Renegotiation
    TB_string= Message[TB_index:SR_index]
    SCIR_index = Message.find("Secure Client-Initiated Renegotiation") #Security Client_ Initated Renegotiation
    SR_string= Message[SR_index:SCIR_index]
    Crime_index = Message.find("CRIME, TLS [m(CVE-2012-4929)")
    SCIR_string= Message[SCIR_index:Crime_index]
    Breach_index = Message.find("BREACH[m (CVE-2013-3587)")
    Crime_string=Message[Crime_index:Breach_index]
    Poodle_index = Message.find("POODLE, SSL[m (CVE-2014-3566)")
    Breach_string= Message[Breach_index:Poodle_index]
    print(Breach_string)
    TLSfallback_index = Message.find("TLS_FALLBACK_SCSV[m (RFC 7507)")
    Poolde_string= Message[Poodle_index:TLSfallback_index]
    sweet_index= Message.find("SWEET32[m (CVE-2016-2183, CVE-2016-6329)")
    TLSfallback_string= Message[TLSfallback_index:sweet_index]
    freak_index= Message.find("FREAK[m (CVE-2015-0204)")
    sweet_string=Message[sweet_index:freak_index]
    drown_index= Message.find("DROWN[m (CVE-2016-0800, CVE-2016-0703)")
    freak_string= Message[freak_index:drown_index]
    Logjam_index= Message.find("LOGJAM[m (CVE-2015-4000), experimental")
    drown_string=Message[drown_index:Logjam_index]
    Beast_index= Message.find("BEAST[m (CVE-2011-3389)")
    Logjam_string= Message[Logjam_index:Beast_index]
    Lucky13_index= Message.find("LUCKY13[m (CVE-2013-0169), experimental")
    Beast_string= Message[Beast_index:Lucky13_index]
    RC3_index = Message.find("RC4[m (CVE-2013-2566, CVE-2015-2808)")
    Lucky13_string= Message[Lucky13_index:RC3_index]
    RC4_string= Message[RC3_index:Message.find("Done")]
    V_string=[HB_string,CSS_string,TB_string,SR_string,SCIR_string,
              Crime_string,Poolde_string,sweet_string,
              freak_string, drown_string,Logjam_string,Beast_string,Lucky13_string,
               RC4_string]

   # V_string={"HeartBleed" : HB_string,
    #          "CSS" : CSS_string,
     #         "Ticket Bleed " : TB_string,
      #        "Secure Renegotiation": SR_string,
       #       "Secure Client-Initiated Renegotiation":SCIR_string,
        #      "Crime": Crime_string,
         #     "Poodle": Poolde_string,
          #    "Sweet": sweet_string,
          #    "Freak" : freak_string,
          #    "Drown" : drown_string,
          #    "Logjam" : Logjam_string,
          #    "Beast" : Beast_string,
           #   "Lucky13" : Lucky13_string,
            #  "RC4" : RC4_string}
    Name_string=["HeartBleed" ,
          "CSS",
          "Ticket Bleed ",
          "Secure Renegotiation",
          "Secure Client-Initiated Renegotiation",
          "Crime",
          "Poodle",
          "Sweet",
          "Freak",
          "Drown",
          "Logjam",
          "Beast",
          "Lucky13",
          "RC4"]
    Vuln = []
    Valchk = 0
    st = Check_Com(Breach_string)
    for x in range(len(st)):
        Vuln.append(st[x])

    st = Check_Com(Breach_string)
    if st[1] != None: # check if all attack not not vulnerable , if valchk = 0 means no need to return value
        Valchk = Valchk +1

    st3 = Chk_TLS(TLSfallback_string)
    for x in range(len(st3)):
        Vuln.append(st3[x])
    if st3[1] != None :
        Valchk = Valchk +1
    print(Vuln)

    for x in range(len(V_string)):
        print(str(x) + V_string[x] + Name_string[x])
        st1=Check_V(Message= V_string[x], Attack=Name_string[x])
        for g in range(len(st1)):
            Vuln.append(st1[g])
        if st1[1] != None:
            Valchk= Valchk +1

        print(Vuln)

    if Valchk != 0:
        Vuln= [IP] + Vuln
        print(Vuln)


def Check_V(Message, Attack):
    if"not vulnerable" in Message:
        r_msg= [
            Attack,
            None,
            None
        ]
        return r_msg
    elif "VULNERABLE" in Message:
        index= Message.find("VULNERABLE")
        comment = Message[index:-1]
        r_msg= [
               Attack,
               "Vulnerable",
               comment
        ]

        return r_msg
    elif "potentially [1;33mVULNERABLE" in Message:
        index = Message.find("potentially [1;33mVULNERABLE")
        comment = Message[index:-1]
        r_msg = [
                  Attack,
                 "Potential Vulnerable",
                 comment
        ]
        return r_msg
    else :
        r_msg = [
            Attack,
            None,
            None
            ]

        return

def Check_Com(Message):
    if "no HTTP compression (OK)" in Message:
        r_msg= [
            "Breach",
             None,
             None
        ]
        return r_msg
    elif len(Message)==0 :
        r_msg=[
            "Breach",
            None,
             None
            ]
        return r_msg

    else :
        comment= Message
        r_msg= [
             "Breach",
             "Vulnerable",
             comment
        ]
        return r_msg
def Chk_TLS(TLS_string):
    if "Downgrade attack prevention supported" in TLS_string:
       r_msg= [
             "TLSFallback",
             None,
             None
        ]
       return r_msg
    elif len(TLS_string)==0:
        r_msg=[
            "TLSFallback",
             None,
             None
            ]
        return r_msg
    else :
        r_msg= [
            "TLSFallback",
            "Vulnerable",
            None
        ]
        return r_msg





scan()
