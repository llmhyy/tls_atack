
import csv
import subprocess
result= subprocess.run(["./testssl.sh", "-U","157.240.13.35:443"],
                       stdout= subprocess.PIPE,
                       stderr = subprocess.PIPE)
y= result.stdout
x= result.stderr
yh=y.decode("utf-8")
def MessageParser (IP, Message):
    Test_string= "Testing vulnerabilities"
    F_index= Message.find(Test_string)
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
    Chk_string = ["Heartbleed[m (CVE-2014-0160)",
                  "CCS[m (CVE-2014-0224)",
                  "Secure Renegotiation [m(CVE-2009-3555)",
                  "Secure Client-Initiated Renegotiation",
                  "CRIME, TLS [m(CVE-2012-4929)",
                  "BREACH[m (CVE-2013-3587)",
                  "POODLE, SSL[m (CVE-2014-3566)",
                  "TLS_FALLBACK_SCSV[m (RFC 7507)",
                  "SWEET32[m (CVE-2016-2183, CVE-2016-6329)",
                  "FREAK[m (CVE-2015-0204)",
                  "DROWN[m (CVE-2016-0800, CVE-2016-0703)",
                  "LOGJAM[m (CVE-2015-4000), experimental",
                  "BEAST[m (CVE-2011-3389)",
                  "LUCKY13[m (CVE-2013-0169), experimental",
                  "RC4[m (CVE-2013-2566, CVE-2015-2808)"]

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

        #data= pd.DataFrame(Vuln)
        #print("DATA" + data)
        #Toexcel= pd.ExcelWriter("ip1.xlsx", engine= "xlsxwriter")
        #data.to_excel(Toexcel, sheet_name= "shee1", startcol= 3)
        #Toexcel.save()
        with open("ip1.csv", "a") as f :
            writer= csv.writer(f,dialect = "excel")
            writer.writerow(Vuln)
        return Vuln
    else:
        return None


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





MessageParser(IP= "157.240.13.35:443", Message=yh)
