from urllib.parse import urlencode, urlparse, parse_qs

from lxml.html import fromstring
from requests import get
import csv
import time 
keyword= "secured"
q_url= "search?q="+keyword
g_string = "https://www.google.com/"
def geturl():
    
    qurl= q_url
    url_list =[]
    list2=[]
    for x in range(0,10,1):
        raw = get(g_string+qurl).text
        page = fromstring(raw)
        url_csv(page)
        nxt_page= page.cssselect("a.fl")
        for g in nxt_page:
            url_list.append(g.get("href"))
           
        for x in range(len(url_list)):
            if "start=" in url_list[x]:
                list2.append(url_list[x])
            
        qurl = list2[-1]
        href= nxt_page[-1]
        href = href.get("href")
        
        time.sleep(5)
    
    

def url_csv(page_info):
    for result in page_info.cssselect(".r a"):
        url = result.get("href")
        if url.startswith("/url?"):
            url = parse_qs(urlparse(url).query)['q']
        print(url[0])
        if "https://" in url[0]:
            with open("urls.csv","a") as f :
                writer= csv.writer(f, dialect="excel")
                writer.writerow([url[0]])
geturl()
