from urllib.parse import urlencode, urlparse, parse_qs
from random_word import RandomWords
from lxml.html import fromstring
from requests import get
import csv
import time
import timeit
import subprocess

domain = {}
g_string = "https://www.google.com/"
times =[]
def updtime (tim):
    if len(times)==0:
        times.append(tim)
    if len(times)!=0:
        print("time initialized")
        init = times[-1]
        diff = init-tim
        times[-1]=diff
        Last_time=time.time()
        with open("time1.csv","a") as g:
            w=csv.writer(g, dialect="excel")
            w.writerow([diff])
        times.append(Last_time)
        print(times)
def testssl(ip1):
    start_time = time.time()
    ip=str(ip1)
    st = "./testssl.sh --csvfile 8.csv --append -U " + ip
    print(st)
    try:
        subprocess.call(st, stdout=subprocess.PIPE,
                        shell=True, timeout=120)
    except:
        print("timeout")
    endtime = start_time-time.time()
    with open("ssltime.csv",'a') as s:
        w= csv.writer(s,dialect='excel')
        w.writerow([endtime])
def search():
    for i in range(100):
        r= RandomWords()
        keyword = r.get_random_word()
        keyword = "https+"+keyword
        q_url1 = "search?q=" + keyword
        geturl(q_url1)
        

def geturl(q_url):

    qurl = q_url

    url_list = []

    list2 = []

    full_url= g_string + qurl

    number=  check_len(full_url)

    if number!= None :

        for x in range(0,number):
            raw = get(g_string + qurl).text
            page = fromstring(raw)
            url_csv(page)
            nxt_page = page.cssselect("a.fl")

            for g in nxt_page:

                url_list.append(g.get("href"))



            for g in range(len(url_list)):

                if "start=" in url_list[g]:



                    list2.append(url_list[g])

            if len(nxt_page) != 0:

                list2.append(url_list[g])

                qurl = list2[-1]

                href = nxt_page[-1]

                href = href.get("href")

            if len(nxt_page)==0:
                return 0 




def url_csv(page_info):
    for result in page_info.cssselect(".r a"):
        url = result.get("href")
        if url.startswith("/url?"):
            url = parse_qs(urlparse(url).query)['q']
        if "https://" in url[0]:
            check_url(url[0] )

def check_url(url):
    start_time=time.time()
    f_string= url
    c_string= f_string[8:-1]
    index=c_string.find("/")
    c_string=c_string[7:index]
    search_dic(c_string, url)
    endtime = start_time-time.time()
    with open("checkurl.csv", "a") as c:
        w= csv.writer(c,dialect='excel')
        w.writerow([endtime])

def search_dic(lookup,url):
    start_time = time.time()
    if domain.get(lookup) ==None:
        with open("full.csv", "a", newline='') as g :
            h = csv.writer(g, dialect="excel")
            h.writerow([url])
            testssl("https://"+lookup)
            update_time=time.time()
            updtime(update_time)
            print("This lookup :")
            print(lookup)
            print("This is URL")
            print(url)
            domain.update({lookup:1 })
            print(domain)
    print(start_time-time.time())

    print(time.time()-start_time)
def check_len(full_url):
    start_time = time.time()
    raw = get(full_url).text
    page = fromstring(raw)
    meta_1= raw.find("""id="resultStats""")
    if meta_1 != -1:
        meta_2= meta_1 +150
        meta = raw[meta_1+23:meta_2]
        meta_1= meta.find("results")
        meta = meta[0:meta_1-1]
        if len(meta)<4 :
            r_msg = len(meta)*7
            return r_msg
        else:
            return 80
    else:

        return None

with open("full.csv", "r") as f:
    r =csv.reader(f, dialect ="excel")
    for g in r:
        if len(g)!=0:
            if domain.get(g[0]) == None:
               domain.update({g[0]: 1})



while True:

    try:
        search()

        print(len(domain))
    except Exception as e:
        print(e)
        print("something went wrong")


