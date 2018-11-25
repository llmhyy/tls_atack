from urllib.parse import urlencode, urlparse, parse_qs
from random_word import RandomWords
from lxml.html import fromstring
from requests import get
import csv
import time

domain = {}
g_string = "https://www.google.com/"
def search():
    for i in range(100):
        r= RandomWords()
        keyword = r.get_random_word()
        keyword = "https+"+keyword
        q_url1 = "search?q=" + keyword
        time.sleep(20)
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
                    print(url_list[g])
                    list2.append(url_list[g])
            if len(list2) != 0:
                list2.append(url_list[g])
                qurl = list2[-1]
                href = nxt_page[-1]
                href = href.get("href")
            dim = 5+(len(domain)*0.003)
            time.sleep(dim)

def url_csv(page_info):
    for result in page_info.cssselect(".r a"):
        url = result.get("href")
        if url.startswith("/url?"):
            url = parse_qs(urlparse(url).query)['q']
        print(url[0])
        if "https://" in url[0]:
            check_url(url[0], )

def check_url(url):

        f_string= url 
        c_string= f_string[8:-1]
        index=c_string.find("/")
        c_string=c_string[0:index]
        search_dic(c_string, url)

def search_dic(lookup,url):
    if domain.get(lookup) ==None:
        with open("refined.csv", "a", newline='') as g :
            h = csv.writer(g, dialect="excel")
            h.writerow([url])
            domain.update({lookup:1 })
            print(domain)
def check_len(full_url):
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
    
                
while len(domain)<10000:
    search()
    print(len(domain))
    


