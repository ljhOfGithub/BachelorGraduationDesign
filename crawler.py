from selenium import webdriver
import time
import requests
from ast import literal_eval
import csv
import xlrd
import yaml
import pandas
import datetime
import numpy
import math
from matplotlib.pyplot import MultipleLocator
from bs4 import BeautifulSoup
apikey = "ZF9TQA39PFPPUD7VCDK2Q9ZVD2M72N2HGZ"
apikey = "P3FE926UGARGQF8HKPM4XWJ38CJAGX5WHZ"
import json
import traceback
import telethon
from telethon import TelegramClient, events
import asyncio
import asyncio.subprocess
from scipy import stats
import numpy as np
import seaborn as sns
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
import random
import networkx as nx
import matplotlib.image as mpimg
import img2pdf
import os
import pandas as pd
import scipy.sparse as sp
import torch
import re
import logging
def testSele():
    logging.basicConfig(level=logging.ERROR, # 控制台打印的日志级别，最低，低于设置的则不输出，如果设置为DEBUG，其他库会输出对应的DEBUG信息
                        filename='tg.log',
                        filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        # a是追加模式，默认如果不写的话，就是追加模式
                        # format=
                        # '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        # 日志格式
                        )
    # tpurl = 'https://bitcointalk.org/index.php?topic=5392806.0'
    # url = 'https://bitcointalk.org/index.php?board=238.40'
    # with open('testpage2.html','w',encoding='utf-8') as f:
    #     print(pageSource,file=f)
    # with open('testpage2.html','r',encoding='utf-8') as f:
    #     pageSource = f.read()
    # print(pageSource)
    chromeOptions = webdriver.ChromeOptions()
    # chromeOptions.add_argument("--proxy-server=https://127.0.0.1:1080")
    driver = webdriver.Chrome(chrome_options=chromeOptions)
    # with open('topic.csv','w',newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['link'])
    #     tplinks = []
    #     for pagenum in range(0,397,4):
    #         try:
    #             bdurl = 'https://bitcointalk.org/index.php?board=238.' + str(pagenum) + '0'
    #             driver.get(bdurl)
    #             time.sleep(5)
    #             pageSource = driver.page_source
    #             bdRe = r'https://bitcointalk.org/index.php\?topic=[0-9]*'#在board中筛选topic的链接
    #             tpGroup = re.findall(bdRe,pageSource)#在每一页board中筛选topic的链接
    #             #每一页都要置空topic链接
    #             for tp in tpGroup:
    #                 if tp.startswith('https://bitcointalk.org/index.php?topic=') and tp != 'https://bitcointalk.org/index.php?topic=':
    #                     tplink = tp[0:40+7]#提取出topic链接
    #                     if tplink not in tplinks:#不重复的topic链接写入csv和列表
    #                         writer.writerow([tplink])
    #                         tplinks.append(tplink)#添加topic链接
    #             tplinks = list(set(tplinks))#去重
    #             # print(tplinks)
    #         except:
    #             traceback.print_exc()
    # return
    df = pd.read_csv('topic.csv')
    haltnum = 3905
    df = df.iloc[haltnum:,]
    filename = 'tglink' + str(haltnum) + '.csv'
    with open(filename,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['link'])
        count = haltnum
        tglinks = []
        for index,row in df.iterrows():
            try:
                print(count)
                count += 1
                tpurl = row['link']
                driver.get(tpurl)
                time.sleep(5)
                pageSource = driver.page_source
                tgRe = r"(https?:\/\/)?(www[.])?(telegram|t)\.me\/([a-zA-Z0-9_-]*)\/?"  # 搜索一个topic下面的telegram链接
                telegroup = re.findall(tgRe, pageSource)
                for link in telegroup:
                    tglink = link[0] + link[2] + '.me/' + link[3]
                    if tglink not in tglinks:
                        print(tglink)
                        tglinks.append(tglink)
                        writer.writerow([tglink])
                # telegroup = [link[0] + link[2] + '.me/' + link[3] for link in telegroup]
                # telegroup = list(set(telegroup))
                # writer.writerow([telegroup])
            except:
                traceback.print_exc()
                print(row['link'])
                logging.error(count)
                # logging.error(str(traceback.format_exc()))
    driver.quit()
    # with open('testpage.html','r',encoding='utf-8') as f:
    #     pageSource = f.read()
    # myRe = 'https://t.me/*'
    #从所有的topic链接中再提取tg链接
    # with open('testpage.html','w',encoding='utf-8') as f:
    #     print(pageSource,file=f)
#337 347 389 431 475 486 537 891 1271 1281 1393 1405 1801 1804 1820 1830 1922 1962 2036 2045
#ERROR:root:2287 2315 2731 2732 2733 2734 2735 2736 2737 2738 2802 2807 2883 2975 2996 3039 3149 3204 3222 3241 3331 3336
#3409 3487 3489 3519 3574 3599 3691 3574 3599 3691 3750 3783 3798 3831 3845
def alltg():
    filedir = os.getcwd()
    filelist = os.listdir(filedir)
    frames = []
    for filename in filelist:
        if filename.startswith('tglink'):
            df = pd.read_csv(filename)
            frames.append(df)
    df = pd.concat(frames)
    df.to_csv('tglinkAll.csv')
#mlnode应该是scamRelNode，和欺诈交易有关的节点，将错就错
#mlnode2应该是正常节点及其一跳节点
def bloxyhack():
    pagenum = 177
    url2type = {}  # 将所有的链接和种类注解存入字典中再进行筛选
    for i in range(1,pagenum+1):
        print(i)
        url = "https://bloxy.info/tokens?annotation_page=" + str(i) + "&blockchain_id=1&q=hack"
        session = requests.session()
        resp = session.get(url)
        html = resp.text
        bs = BeautifulSoup(html,features="lxml")
        tbody = bs.select('body > div.wrapper > div:nth-child(5) > div:nth-child(8) > div > div > div.panel-body > div.table-responsive > table > tbody')[0]
        for tr in tbody:
            for tag in tr:
                if not isinstance(tag, str):#排除空行换行符等特殊的tag
                    if tag.select("a"):
                        atag = tag.select("a")[0]
                        url = atag.get('href')
                        mytype = atag.string
                        url2type[url] = mytype
        # print(url2type)
    with open('bloxyhack.txt','w',encoding='utf-8') as f:
        print(url2type,file=f)
    json_str = json.dumps(url2type)
    with open('bloxyhack.json', 'w') as json_file:
        json_file.write(json_str)
def bloxyscam():
    pagenum = 56
    url2type = {}  # 将所有的链接和种类注解存入字典中再进行筛选
    for i in range(1,pagenum+1):
        print(i)
        url = "https://bloxy.info/tokens?annotation_page=" + str(i) + "&blockchain_id=1&q=hack"
        session = requests.session()
        resp = session.get(url)
        html = resp.text
        bs = BeautifulSoup(html,features="lxml")
        tbody = bs.select('body > div.wrapper > div:nth-child(5) > div:nth-child(8) > div > div > div.panel-body > div.table-responsive > table > tbody')[0]
        for tr in tbody:
            for tag in tr:
                if not isinstance(tag, str):#排除空行换行符等特殊的tag
                    if tag.select("a"):
                        atag = tag.select("a")[0]
                        url = atag.get('href')
                        mytype = atag.string
                        url2type[url] = mytype
    with open('bloxyscam.txt','w',encoding='utf-8') as f:
        print(url2type,file=f)
    json_str = json.dumps(url2type)
    with open('bloxyscam.json', 'w') as json_file:
        json_file.write(json_str)
def bloxymalware():
    pagenum = 1
    url2type = {}  # 将所有的链接和种类注解存入字典中再进行筛选
    for i in range(1,pagenum+1):
        print(i)
        url = "https://bloxy.info/tokens?annotation_page=" + str(i) + "&blockchain_id=1&q=hack"
        session = requests.session()
        resp = session.get(url)
        html = resp.text
        bs = BeautifulSoup(html,features="lxml")
        tbody = bs.select('body > div.wrapper > div:nth-child(5) > div:nth-child(8) > div > div > div.panel-body > div.table-responsive > table > tbody')[0]
        for tr in tbody:
            for tag in tr:
                if not isinstance(tag, str):#排除空行换行符等特殊的tag
                    if tag.select("a"):
                        atag = tag.select("a")[0]
                        url = atag.get('href')
                        mytype = atag.string
                        url2type[url] = mytype
    with open('bloxymalware.txt','w',encoding='utf-8') as f:
        print(url2type,file=f)
    json_str = json.dumps(url2type)
    with open('bloxymalware.json', 'w') as json_file:
        json_file.write(json_str)

def bloxy2():
    with open('bloxyhack.txt','r',encoding='utf-8') as f:
        dict1 = literal_eval(f.read())
    with open('bloxyscam.txt','r',encoding='utf-8') as f:
        dict2 = literal_eval(f.read())
    with open('bloxymalware.txt','r',encoding='utf-8') as f:
        dict3 = literal_eval(f.read())
    mydict = {}
    mydict.update(dict1)
    mydict.update(dict2)
    mydict.update(dict3)
    newdict = {}
    for key,value in mydict.items():
        newdict[key.lower()] = value.lower()
    mydict = newdict
    # print(mydict)
    hacktype = ['Phish', 'Trust-Trading', 'Hack', 'Malware', 'Upbit', 'Scam', 'hackethereumIco']
    notType = ['Hacking', 'HackToken', 'HackDao', 'HackerGold', 'Hacken', 'HackerSpaceBarneysToken', 'HackerToken', 'Gitcoin', 'VitaluckHack']
    # hacktype = 'PhishTrust-TradingTrust TradingHackMalwareUpbitScam'
    addrlist = []
    # notType = 'HackingHackTokenHackDaoHackerGoldHackenHackerSpaceBarneysToken'
    for url,type in mydict.items():
        for key in hacktype:
            if key.lower() in type.lower():
                if url.startswith('/address'):
                    addr = url[9:]
                elif url.startswith('/tx'):
                    addr = url[4:]
                if len(addr) == 42:
                    addrlist.append(addr)
    removelist = []
    for addr in addrlist:
        for nottag in notType:
            if '/tx/'+str(addr) in mydict.keys():
                if nottag.lower() in mydict['/tx/'+str(addr)].lower():
                    removelist.append(addr)
            elif '/address/'+str(addr) in mydict.keys():
                if nottag.lower() in mydict['/address/'+str(addr)].lower():
                    removelist.append(addr)
    for removeaddr in removelist:
        addrlist.remove(removeaddr)
    print(removelist)
    print(addrlist)
    addrlist = list(set(addrlist))
    # print(addrlist)
    print(len(addrlist))
    list1 = list(set(dict1.values()))
    with open('bloxyAddr.txt','w',encoding='utf-8') as f:
        print(addrlist,file=f)
    # print(list1)
    # print(len(dict1))
def differSetNtx():
    addrlist = deduplicate()
    session = requests.Session()
    with open('bntx1.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber", "timeStamp", "hash", "nonce", "blockHash", "transactionIndex",
                         "from", "to", "value", "gas", "gasPrice", "isError", "txreceipt_status", "input",
                         "contractAddress", "cumulativeGasUsed", "gasUsed", "confirmations"])
        number = 0
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlist&address=" + addr + "&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'], result['timeStamp'], result['hash'], result['nonce'],
                                         result['blockHash'],
                                         result['transactionIndex'], result['from'], result['to'], result['value'],
                                         result['gas'], result['gasPrice'],
                                         result['isError'], result['txreceipt_status'], result['input'],
                                         result['contractAddress'], result['cumulativeGasUsed'],
                                         result['gasUsed'], result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def differSetItx():
    addrlist = deduplicate()
    session = requests.Session()
    with open('bitx1.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","from","to","value",
                         "contractAddress","input","type","gas","gasUsed","traceId",
                         "isError","errCode"])
        number = 0
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlistinternal&address=" + addr + "&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            print(url)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['from'],
                                         result['to'],result['value'],result['contractAddress'],result['input'],result['type'],
                                         result['gas'],result['gasUsed'],result['traceId'],result['isError'],result['errCode']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)

def differSetEtx():
    addrlist = deduplicate()
    session = requests.Session()
    with open('betx1.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","from",
                         "contractAddress","to","value","tokenName","tokenSymbol","tokenDecimal",
                         "transactionIndex","gas","gasPrice","gasUsed","cumulativeGasUsed","input","confirmations"])
        number = 0
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=tokentx&address=" + addr + "&page=1&offset=10000&startblock=0&endblock=99999999&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],
                                         result['blockHash'],result['from'],result['contractAddress'],result['to'],result['value'],
                                         result['tokenName'],result['tokenSymbol'],result['tokenDecimal'],result['transactionIndex'],result['gas'],
                                         result['gasPrice'],result['gasUsed'],result['cumulativeGasUsed'],result['input'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def deduplicate():
    with open('bitpoint.txt','r') as f:
        list1 = literal_eval(f.read())
        list1 = [addr.lower() for addr in list1]
    with open('cryptopia.txt','r') as f:
        list2 = literal_eval(f.read())
        list2 = [addr.lower() for addr in list2]
    with open('etherdelta.txt','r') as f:
        list3 = literal_eval(f.read())
        list3 = [addr.lower() for addr in list3]
    with open('LendfMe.txt','r') as f:
        list4 = literal_eval(f.read())
        list4 = [addr.lower() for addr in list4]
    with open('phishaddr.txt','r') as f:
        list5 = literal_eval(f.read())
        list5 = [addr.lower() for addr in list5]
    with open('upbit.txt','r') as f:
        list6 = literal_eval(f.read())
        list6 = [addr.lower() for addr in list6]
    with open('scam.txt','r') as f:
        list7 = literal_eval(f.read())
        list7 = [addr.lower() for addr in list7]
    with open('db.txt','r') as f:
        list8 = literal_eval(f.read())
        list8 = [addr.lower() for addr in list8]
    with open('bloxyAddr.txt','r') as f:
        list9 = literal_eval(f.read())
        list9 = [addr.lower() for addr in list9]
    # print(type(list1))
    alladdr1 = list(set(list1 + list2 + list3 + list4 + list5 + list6 + list7 + list8 + list9))
    alladdr2 = list(set(list1 + list2 + list3 + list4 + list5 + list6 + list7 + list8))
    ret = [i for i in alladdr1 if i not in alladdr2]
    print(ret)
    print(len(alladdr1))
    # return ret
    with open('addr.txt','w',encoding='utf-8') as f:
        print(alladdr1,file=f)
def readxlsx():
    workbook = xlrd.open_workbook('scam.xlsx')
    worksheet = workbook.sheet_by_index(0)
    objlist = worksheet.col_values(2)[1:]
    addrlist = []
    for obj in objlist:
            if len(obj) > 42:
                splitlist = obj.split(",")
                for addr in splitlist:
                    if addr.startswith('0x'):
                        addrlist.append(addr)
    with open('scam.txt','w',encoding='utf-8') as f:
        print(addrlist,file=f)

def writedb():
    with open('urls.yaml','r',encoding='utf-8') as f:
        with open('dblist.txt','w',encoding='utf-8')as f2:
            with open('temp.txt','w',encoding='utf-8') as f3:
                temp = yaml.load(f.read(),Loader=yaml.FullLoader)
                print(temp,file=f2)
                for item in temp:
                    f3.write(str(item) + '\n')
    with open('dblist.txt','r',encoding='utf-8') as f:
        itemlist = literal_eval(f.read())
        print(type(itemlist))
def readdb():
    with open('dblist.txt','r',encoding='utf-8') as f:
        itemlist = literal_eval(f.read())
        addrlist = []
        for item in itemlist:
            if 'addresses' in item.keys() and 'ETH' in item['addresses']:
                    addrs = item['addresses']['ETH']
                    for addr in addrs:
                        if len(addr) == 42:
                            addrlist.append(addr)
    addrlist = [addr.lower() for addr in addrlist]
    addrlist = list(set(addrlist))
    print(len(addrlist))
    with open('db.txt','w',encoding='utf-8') as f:
        print(addrlist,file=f)

def getNtxs1():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[0:1500]
    session = requests.Session()
    with open('ntx1.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","transactionIndex",
                         "from","to","value","gas","gasPrice","isError","txreceipt_status","input",
                         "contractAddress","cumulativeGasUsed","gasUsed","confirmations"])
        number = 0
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlist&address="+addr+"&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],result['blockHash'],
                                        result['transactionIndex'],result['from'],result['to'],result['value'],result['gas'],result['gasPrice'],
                                        result['isError'],result['txreceipt_status'],result['input'],result['contractAddress'],result['cumulativeGasUsed'],
                                        result['gasUsed'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getNtxs2():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[1500:3000]
    session = requests.Session()
    with open('ntx2.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","transactionIndex",
                         "from","to","value","gas","gasPrice","isError","txreceipt_status","input",
                         "contractAddress","cumulativeGasUsed","gasUsed","confirmations"])
        number = 1500
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlist&address="+addr+"&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],result['blockHash'],
                                        result['transactionIndex'],result['from'],result['to'],result['value'],result['gas'],result['gasPrice'],
                                        result['isError'],result['txreceipt_status'],result['input'],result['contractAddress'],result['cumulativeGasUsed'],
                                        result['gasUsed'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getNtxs3():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[3000:4500]
    session = requests.Session()
    with open('ntx3.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","transactionIndex",
                         "from","to","value","gas","gasPrice","isError","txreceipt_status","input",
                         "contractAddress","cumulativeGasUsed","gasUsed","confirmations"])
        number = 3000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlist&address="+addr+"&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],result['blockHash'],
                                        result['transactionIndex'],result['from'],result['to'],result['value'],result['gas'],result['gasPrice'],
                                        result['isError'],result['txreceipt_status'],result['input'],result['contractAddress'],result['cumulativeGasUsed'],
                                        result['gasUsed'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)

def getNtxs4():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[4500:]
    session = requests.Session()
    with open('ntx4.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","transactionIndex",
                         "from","to","value","gas","gasPrice","isError","txreceipt_status","input",
                         "contractAddress","cumulativeGasUsed","gasUsed","confirmations"])
        number = 4500
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlist&address="+addr+"&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],result['blockHash'],
                                        result['transactionIndex'],result['from'],result['to'],result['value'],result['gas'],result['gasPrice'],
                                        result['isError'],result['txreceipt_status'],result['input'],result['contractAddress'],result['cumulativeGasUsed'],
                                        result['gasUsed'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
# def remointeraction():
#     with open('ntx1.csv','r') as f:
#         ntx1 = literal_eval()
def getItxs1():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[0:1500]
    session = requests.Session()
    with open('itx1.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","from","to","value",
                         "contractAddress","input","type","gas","gasUsed","traceId",
                         "isError","errCode"])
        number = 0
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlistinternal&address=" + addr + "&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            print(url)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['from'],
                                         result['to'],result['value'],result['contractAddress'],result['input'],result['type'],
                                         result['gas'],result['gasUsed'],result['traceId'],result['isError'],result['errCode']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getItxs2():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[1500:3000]
    session = requests.Session()
    with open('itx2.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","from","to","value",
                         "contractAddress","input","type","gas","gasUsed","traceId",
                         "isError","errCode"])
        number = 1500
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlistinternal&address=" + addr + "&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            print(url)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['from'],
                                         result['to'],result['value'],result['contractAddress'],result['input'],result['type'],
                                         result['gas'],result['gasUsed'],result['traceId'],result['isError'],result['errCode']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getItxs3():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[3000:4500]
    session = requests.Session()
    with open('itx3.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","from","to","value",
                         "contractAddress","input","type","gas","gasUsed","traceId",
                         "isError","errCode"])
        number = 3000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlistinternal&address=" + addr + "&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            print(url)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['from'],
                                         result['to'],result['value'],result['contractAddress'],result['input'],result['type'],
                                         result['gas'],result['gasUsed'],result['traceId'],result['isError'],result['errCode']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)

def getItxs4():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[4500:]
    session = requests.Session()
    with open('itx4.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","from","to","value",
                         "contractAddress","input","type","gas","gasUsed","traceId",
                         "isError","errCode"])
        number = 4500
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlistinternal&address=" + addr + "&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            print(url)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['from'],
                                         result['to'],result['value'],result['contractAddress'],result['input'],result['type'],
                                         result['gas'],result['gasUsed'],result['traceId'],result['isError'],result['errCode']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)

def getEtxs1():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[0:1500]
    session = requests.Session()
    with open('etx1.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","from",
                         "contractAddress","to","value","tokenName","tokenSymbol","tokenDecimal",
                         "transactionIndex","gas","gasPrice","gasUsed","cumulativeGasUsed","input","confirmations"])
        number = 0
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=tokentx&address=" + addr + "&page=1&offset=10000&startblock=0&endblock=99999999&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],
                                         result['blockHash'],result['from'],result['contractAddress'],result['to'],result['value'],
                                         result['tokenName'],result['tokenSymbol'],result['tokenDecimal'],result['transactionIndex'],result['gas'],
                                         result['gasPrice'],result['gasUsed'],result['cumulativeGasUsed'],result['input'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getEtxs2():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[1500:3000]
    session = requests.Session()
    with open('etx2.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","from",
                         "contractAddress","to","value","tokenName","tokenSymbol","tokenDecimal",
                         "transactionIndex","gas","gasPrice","gasUsed","cumulativeGasUsed","input","confirmations"])
        number = 1500
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=tokentx&address=" + addr + "&page=1&offset=10000&startblock=0&endblock=99999999&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],
                                         result['blockHash'],result['from'],result['contractAddress'],result['to'],result['value'],
                                         result['tokenName'],result['tokenSymbol'],result['tokenDecimal'],result['transactionIndex'],result['gas'],
                                         result['gasPrice'],result['gasUsed'],result['cumulativeGasUsed'],result['input'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getEtxs3():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[3000:4500]
    session = requests.Session()
    with open('etx3.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","from",
                         "contractAddress","to","value","tokenName","tokenSymbol","tokenDecimal",
                         "transactionIndex","gas","gasPrice","gasUsed","cumulativeGasUsed","input","confirmations"])
        number = 3000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=tokentx&address=" + addr + "&page=1&offset=10000&startblock=0&endblock=99999999&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],
                                         result['blockHash'],result['from'],result['contractAddress'],result['to'],result['value'],
                                         result['tokenName'],result['tokenSymbol'],result['tokenDecimal'],result['transactionIndex'],result['gas'],
                                         result['gasPrice'],result['gasUsed'],result['cumulativeGasUsed'],result['input'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)

def getEtxs4():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[4500:]
    session = requests.Session()
    with open('etx4.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","from",
                         "contractAddress","to","value","tokenName","tokenSymbol","tokenDecimal",
                         "transactionIndex","gas","gasPrice","gasUsed","cumulativeGasUsed","input","confirmations"])
        number = 4500
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=tokentx&address=" + addr + "&page=1&offset=10000&startblock=0&endblock=99999999&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],
                                         result['blockHash'],result['from'],result['contractAddress'],result['to'],result['value'],
                                         result['tokenName'],result['tokenSymbol'],result['tokenDecimal'],result['transactionIndex'],result['gas'],
                                         result['gasPrice'],result['gasUsed'],result['cumulativeGasUsed'],result['input'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
# 代币：https://api.etherscan.io/api?module=account&action=tokentx&address=0x0f3257e9513f4812bf015efc5022f16bfef0cfa8&page=1&offset=100&startblock=0&endblock=27025780&sort=asc&apikey=YourApiKeyToken
# 一般：https://api.etherscan.io/api?module=account&action=txlist&address=0xd0b0d5a8c0b40b7272115a23a2d5e36ad190f13c&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=YourApiKeyToken
# 内部：https://api.etherscan.io/api?module=account&action=txlistinternal&address=0xe81ad9c0b999222bfe4060ef36b9fef995bad9f9&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=ZF9TQA39PFPPUD7VCDK2Q9ZVD2M72N2HGZ
#fig2:每个地址首次欺诈交易的时间
#fig3：每个月不同种交易的欺诈交易的数量
#fig5：2019年最多欺诈交易的地址，取出来分析交易数量
#fig6：不同种交易地址的存活时间
#fig7：欺诈地址的一般交易和内部交易的数量分布
def fig2():
    # A = [1,3,2]
    # B = [4,5,6]
    # zipped = zip(A, B)
    # sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    # result = zip(*sort_zipped)
    # x_axis, y_axis = [list(x) for x in result]
    # print(x_axis)
    # print(y_axis)
    # return
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    #找每个地址第一次收到钱的时间戳，包括一般交易和内部交易
    # df = pandas.concat(frames)
    # # df = df.drop_duplicates()#去重
    # df1 = pandas.read_csv('ntx.csv')
    # df2 = pandas.read_csv('itx.csv')
    # frames = [df1,df2]
    # df = pandas.concat(frames)
    # df['timeStamp'] = df['timeStamp'].map(lambda x:datetime.datetime.fromtimestamp(x).strftime("%Y-%m"))
    # df = df.sort_values('timeStamp')
    # addr2time = {}
    # for addr in addrlist:
    #     addr2time[addr] = []
    # for index, row in df.iterrows():
    #     if isinstance(row['to'],str) and row['to'] in addrlist:
    #         addr2time[row['to']].append(row['timeStamp'])
    # for addr,time in addr2time.items():
    #     time.sort()
    # addr2mon = {}
    # for addr,time in addr2time.items():
    #     if len(time) > 0:
    #         addr2mon[addr] = time[0]
    # # print(addr2mon)
    # with open('fig2addr2mon.txt','w',encoding='utf-8') as f:#只存储了欺诈地址对应的月份，没存储unknown节点对应的月份
    #     print(addr2mon,file=f)
    with open('fig2addr2mon.txt','r',encoding='utf-8') as f:
        addr2mon = literal_eval(f.read())
    mon2count = {}
    for addr,mon in addr2mon.items():
        mon2count[mon] = mon2count.get(mon,0) + 1
    # # print(mon2count)
    # mon2count = sorted(mon2count.items(),key = lambda x:(x[0],x[1]))
    # with open('fig2mon2count.txt','w',encoding='utf-8') as f:
    #     print(mon2count,file=f)
    # return
    # with open('fig2mon2count.txt','r',encoding='utf-8') as f:
    #     mon2count = literal_eval(f.read())
    x = mon2count.keys()
    y = mon2count.values()
    zipped = zip(x, y)
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    print(result)
    plt.figure(figsize=(6, 6))
    plt.plot(x,y)
    plt.xticks(rotation=30)
    x_major_locator = MultipleLocator(4)
    ax = plt.gca()
    ax.set_ylabel('# of addrs')
    ax.set_xlabel('Date')
    plt.yscale('log')
    ax.xaxis.set_major_locator(x_major_locator)
    plt.savefig('fig2.jpg',bbox_inches = 'tight')
    plt.show()
def tab1n():#不同种地址的各种交易数量，没有每个地址三种交易的数量
    with open('typelist.txt', 'r') as f:
        typelist = literal_eval(f.read())
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    Malwarelist = typelist['Malware']
    ICOlist = typelist['ICO']
    TrustTradinglist = typelist['Trust-Trading']
    Phishlist = typelist['Phish']
    Hacklist = typelist['Hack']
    notMalwarelist = [addr for addr in addrlist if addr not in Malwarelist]
    notICOlist = [addr for addr in addrlist if addr not in ICOlist]
    notTrustTradinglist = [addr for addr in addrlist if addr not in TrustTradinglist]
    notPhishlist = [addr for addr in addrlist if addr not in Phishlist]
    notHacklist = [addr for addr in addrlist if addr not in Hacklist]
    df = pandas.read_csv('ntx.csv')
    #统计每个地址的一般交易数量，根据每个地址的入度计算
    typenum = {}
    for type,addrlist in typelist.items():
        typenum[type] = 0
    for index, row in df.iterrows():
        if row['to'] in Malwarelist and row['from'] not in notMalwarelist:
            typenum['Malware'] += 1
        if row['from'] in Malwarelist and row['to'] not in notMalwarelist:
            typenum['Malware'] += 1

        if row['to'] in ICOlist and row['from'] not in notICOlist:
            typenum['ICO'] += 1
        if row['from'] in ICOlist and row['to'] not in notICOlist:
            typenum['ICO'] += 1

        if row['to'] in TrustTradinglist and row['from'] not in notTrustTradinglist:
            typenum['Trust-Trading'] += 1
        if row['from'] in TrustTradinglist and row['to'] not in notTrustTradinglist:
            typenum['Trust-Trading'] += 1

        if row['to'] in Phishlist and row['from'] not in notPhishlist:
            typenum['Phish'] += 1
        if row['from'] in Phishlist and row['to'] not in notPhishlist:
            typenum['Phish'] += 1

        if row['to'] in Hacklist and row['from'] not in notHacklist:
            typenum['Hack'] += 1
        if row['from'] in Hacklist and row['to'] not in notHacklist:
            typenum['Hack'] += 1

    with open('tab1n.txt', 'w') as f:
        print(typenum,file=f)

def tab1i():
    with open('typelist.txt', 'r') as f:
        typelist = literal_eval(f.read())
    with open('addr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    Malwarelist = typelist['Malware']
    ICOlist = typelist['ICO']
    TrustTradinglist = typelist['Trust-Trading']
    Phishlist = typelist['Phish']
    Hacklist = typelist['Hack']
    notMalwarelist = [addr for addr in addrlist if addr not in Malwarelist]
    notICOlist = [addr for addr in addrlist if addr not in ICOlist]
    notTrustTradinglist = [addr for addr in addrlist if addr not in TrustTradinglist]
    notPhishlist = [addr for addr in addrlist if addr not in Phishlist]
    notHacklist = [addr for addr in addrlist if addr not in Hacklist]
    df = pandas.read_csv('itx.csv')
    typenum = {}
    for type, addrlist in typelist.items():
        typenum[type] = 0
    for index, row in df.iterrows():
        if row['to'] in Malwarelist and row['from'] not in notMalwarelist:
            typenum['Malware'] += 1
        if row['from'] in Malwarelist and row['to'] not in notMalwarelist:
            typenum['Malware'] += 1

        if row['to'] in ICOlist and row['from'] not in notICOlist:
            typenum['ICO'] += 1
        if row['from'] in ICOlist and row['to'] not in notICOlist:
            typenum['ICO'] += 1

        if row['to'] in TrustTradinglist and row['from'] not in notTrustTradinglist:
            typenum['Trust-Trading'] += 1
        if row['from'] in TrustTradinglist and row['to'] not in notTrustTradinglist:
            typenum['Trust-Trading'] += 1

        if row['to'] in Phishlist and row['from'] not in notPhishlist:
            typenum['Phish'] += 1
        if row['from'] in Phishlist and row['to'] not in notPhishlist:
            typenum['Phish'] += 1

        if row['to'] in Hacklist and row['from'] not in notHacklist:
            typenum['Hack'] += 1
        if row['from'] in Hacklist and row['to'] not in notHacklist:
            typenum['Hack'] += 1

    with open('tab1i.txt', 'w') as f:
        print(typenum, file=f)

def tab1e():
    with open('typelist.txt', 'r') as f:
        typelist = literal_eval(f.read())
    with open('addr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    Malwarelist = typelist['Malware']
    ICOlist = typelist['ICO']
    TrustTradinglist = typelist['Trust-Trading']
    Phishlist = typelist['Phish']
    Hacklist = typelist['Hack']
    notMalwarelist = [addr for addr in addrlist if addr not in Malwarelist]
    notICOlist = [addr for addr in addrlist if addr not in ICOlist]
    notTrustTradinglist = [addr for addr in addrlist if addr not in TrustTradinglist]
    notPhishlist = [addr for addr in addrlist if addr not in Phishlist]
    notHacklist = [addr for addr in addrlist if addr not in Hacklist]
    df = pandas.read_csv('etx.csv')
    typenum = {}
    for type, addrlist in typelist.items():
        typenum[type] = 0
    for index, row in df.iterrows():
        if row['to'] in Malwarelist and row['from'] not in notMalwarelist:
            typenum['Malware'] += 1
        if row['from'] in Malwarelist and row['to'] not in notMalwarelist:
            typenum['Malware'] += 1

        if row['to'] in ICOlist and row['from'] not in notICOlist:
            typenum['ICO'] += 1
        if row['from'] in ICOlist and row['to'] not in notICOlist:
            typenum['ICO'] += 1

        if row['to'] in TrustTradinglist and row['from'] not in notTrustTradinglist:
            typenum['Trust-Trading'] += 1
        if row['from'] in TrustTradinglist and row['to'] not in notTrustTradinglist:
            typenum['Trust-Trading'] += 1

        if row['to'] in Phishlist and row['from'] not in notPhishlist:
            typenum['Phish'] += 1
        if row['from'] in Phishlist and row['to'] not in notPhishlist:
            typenum['Phish'] += 1

        if row['to'] in Hacklist and row['from'] not in notHacklist:
            typenum['Hack'] += 1
        if row['from'] in Hacklist and row['to'] not in notHacklist:
            typenum['Hack'] += 1

    with open('tab1e.txt', 'w') as f:
        print(typenum, file=f)
def tab1noraddr():
    df = pandas.read_csv('normalAddrntx2.csv')
    print(len(df))
    df = pandas.read_csv('normalAddritx2.csv')
    print(len(df))
    df = pandas.read_csv('normalAddretx2.csv')
    print(len(df))

def nfig3():
    #统计每个月的欺诈交易数量，取出每行的时间戳进行转换，判断时间戳的月份，月份的欺诈交易数量+1
    # df1 = pandas.read_csv('ntx1.csv')
    # df2 = pandas.read_csv('ntx2.csv')
    # df3 = pandas.read_csv('ntx3.csv')
    # df4 = pandas.read_csv('ntx4.csv')
    # df5 = pandas.read_csv('bntx1.csv')
    # frames = [df1,df2,df3,df4,df5]
    # df = pandas.concat(frames)
    df = pandas.read_csv('ntx.csv')
    df['timeStamp'] = df['timeStamp'].map(lambda x:datetime.datetime.fromtimestamp(x).strftime("%Y-%m"))
    dfsort = df.sort_values('timeStamp')
    month2count = {}
    for index,row in dfsort.iterrows():#桶计数，如果当前月份还没有对应的字典则初始化为0，如果已经有对应的字典则取出对应的数量然后在此基础上加1
        month2count[row['timeStamp']] = month2count.get(row['timeStamp'],0) + 1
    print(month2count)
    x = month2count.keys()
    y = month2count.values()
    with open('fig3N.txt','w') as f:
        print(month2count,file=f)
    return
    plt.plot(x,y)
    plt.xticks(rotation=30)
    x_major_locator = MultipleLocator(3)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.yscale('log')
    plt.show()

def ifig3():
    #统计每个月的欺诈交易数量，取出每行的时间戳进行转换，判断时间戳的月份，月份的欺诈交易数量+1
    # df1 = pandas.read_csv('itx1.csv')
    # df2 = pandas.read_csv('itx2.csv')
    # df3 = pandas.read_csv('itx3.csv')
    # df4 = pandas.read_csv('itx4.csv')
    # df5 = pandas.read_csv('bitx1.csv')
    # frames = [df1,df2,df3,df4]
    # df = pandas.concat(frames)
    # df = df.drop_duplicates()
    df = pandas.read_csv('itx.csv')
    df['timeStamp'] = df['timeStamp'].map(lambda x:datetime.datetime.fromtimestamp(x).strftime("%Y-%m"))
    dfsort = df.sort_values('timeStamp')
    month2count = {}
    for index,row in dfsort.iterrows():#桶计数，如果当前月份还没有对应的字典则初始化为0，如果已经有对应的字典则取出对应的数量然后在此基础上加1
        month2count[row['timeStamp']] = month2count.get(row['timeStamp'],0) + 1
    print(month2count)
    x = month2count.keys()
    y = month2count.values()
    with open('fig3I.txt','w') as f:
        print(month2count,file=f)
    return
    plt.plot(x,y)
    plt.xticks(rotation=30)
    x_major_locator = MultipleLocator(3)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.yscale('log')
    plt.show()

def efig3():
    #统计每个月的欺诈交易数量，取出每行的时间戳进行转换，判断时间戳的月份，月份的欺诈交易数量+1
    # df1 = pandas.read_csv('etx1.csv')
    # df2 = pandas.read_csv('etx2.csv')
    # df3 = pandas.read_csv('etx3.csv')
    # df4 = pandas.read_csv('etx4.csv')
    # df5 = pandas.read_csv('betx1.csv')
    # print(df1.shape[0])
    # print(df2.shape[0])
    # print(df3.shape[0])
    # print(df4.shape[0])
    # frames = [df1,df2,df3,df4]
    # df = pandas.concat(frames)
    df = pandas.read_csv('etx.csv')
    # print(df.shape[0])
    # df = df.drop_duplicates()
    # print(df.shape[0])
    df['timeStamp'] = df['timeStamp'].map(lambda x:datetime.datetime.fromtimestamp(x).strftime("%Y-%m"))
    dfsort = df.sort_values('timeStamp')
    month2count = {}
    for index,row in dfsort.iterrows():#桶计数，如果当前月份还没有对应的字典则初始化为0，如果已经有对应的字典则取出对应的数量然后在此基础上加1
        month2count[row['timeStamp']] = month2count.get(row['timeStamp'],0) + 1
    print(month2count)
    x = month2count.keys()
    y = month2count.values()
    with open('fig3E.txt','w') as f:
        print(month2count,file=f)
    return
    plt.plot(x,y)
    plt.xticks(rotation=30)
    x_major_locator = MultipleLocator(3)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.yscale('log')
    plt.show()
def mixfig3():
    with open('fig3N.txt','r') as f:
        month2count1 = literal_eval(f.read())
    with open('fig3I.txt','r') as f:
        month2count2 = literal_eval(f.read())
    with open('fig3E.txt','r') as f:
        month2count3 = literal_eval(f.read())
    x1 = month2count1.keys()
    y1 = month2count1.values()
    x2 = month2count2.keys()
    y2 = month2count2.values()
    x3 = month2count3.keys()
    y3 = month2count3.values()
    plt.figure(figsize=(6, 6))
    plt.plot(x1, y1, label='normal tx', ls='--', c='black')
    plt.plot(x2, y2, label='internal tx', ls='-.', c='black')
    plt.plot(x3, y3, label='token tx', ls='-', c='black')
    plt.xticks(rotation=30)
    x_major_locator = MultipleLocator(4)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.set_ylabel('# of txs')
    ax.set_xlabel('Month')
    plt.yscale('log')
    plt.legend()
    plt.savefig('fig3.jpg',bbox_inches = 'tight')
    plt.show()
def fig221():
    with open('fig2addr2mon.txt','r',encoding='utf-8') as f:
        addr2mon = literal_eval(f.read())
    addrlist21 = []
    for addr,month in addr2mon.items():
        if month[0:4] == '2021':
            addrlist21.append(addr)
    with open('fig221.txt','w',encoding='utf-8') as f:
        print(addrlist21,file=f)
def scamIOcome21():
    with open('fig221.txt','r',encoding='utf-8') as f:
        addrlist21 = literal_eval(f.read())
    with open('scamIncome.txt','r',encoding='utf-8') as f:
        scamIncome = literal_eval(f.read())
    with open('scamOutcome.txt','r',encoding='utf-8') as f:
        scamOutcome = literal_eval(f.read())
    scamIncome21 = {}
    scamOutcome21 = {}
    profit21 = {}
    maxprofit = 0
    for addr in addrlist21:
        if addr in scamIncome.keys():
            scamIncome21[addr] = scamIncome[addr]
            if scamIncome21[addr] > maxprofit:
                maxprofit = scamIncome21[addr]
            if scamIncome21[addr] > 100:
                profit21[addr] = scamIncome21[addr]
        if addr in scamIncome.keys():
            scamOutcome21[addr] = scamOutcome[addr]
    with open('scamIncome21.txt','w') as f:
        print(scamIncome21,file=f)
    with open('scamOutcome21.txt','w') as f:
        print(scamOutcome21,file=f)
#fig4：所有地址每个月欺诈交易的数量，根据txhash统计，存储to地址和txhash的列表的字典，然后去重，统计所有种类的交易
def fig4():#每天的交易数量取最大值
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    # df1 = pandas.read_csv('ntx1.csv')
    # df2 = pandas.read_csv('ntx2.csv')
    # df3 = pandas.read_csv('ntx3.csv')
    # df4 = pandas.read_csv('ntx4.csv')
    # df5 = pandas.read_csv('itx1.csv')
    # df6 = pandas.read_csv('itx2.csv')
    # df7 = pandas.read_csv('itx3.csv')
    # df8 = pandas.read_csv('itx4.csv')
    # df9 = pandas.read_csv('etx1.csv')
    # df10 = pandas.read_csv('etx2.csv')
    # df11 = pandas.read_csv('etx3.csv')
    # df12 = pandas.read_csv('etx4.csv')
    # df13 = pandas.read_csv('bntx1.csv')
    # df14 = pandas.read_csv('bitx1.csv')
    # df15 = pandas.read_csv('betx1.csv')
    # frames = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15]
    # frames = [df1,df2,df3,df4]
    # df = pandas.concat(frames)
    df = pandas.read_csv('ntx.csv')
    # df = df.drop_duplicates()#去重
    # df['timeStamp'] = df['timeStamp'].map(lambda x:datetime.datetime.fromtimestamp(x).strftime("%Y-%m-%d"))
    # df = df.sort_values('timeStamp')
    # pandas.set_option('display.max_columns', 40)  # 打印最大列数
    #月份2地址2交易数量，不同地址作为同x不同y的点
    #地址2月份还是月份2地址都可以，地址2月份2交易数量方便
    #先排序时间再统计交易数量
    #一分钟内的交易数量，取每天的最大值
    # day2addr2num = {}
    # for index, row in df.iterrows():
    #     if isinstance(row['to'],str) and row['to'] in addrlist:
    #         day2addr2num[row['timeStamp']] = {}
    # for index, row in df.iterrows():
    #     if isinstance(row['to'],str) and row['to'] in addrlist:
    #         day2addr2num[row['timeStamp']][row['to']] = day2addr2num[row['timeStamp']].get(row['to'],0) + 1
    # with open('fig4.txt','w',encoding='utf-8') as f:
    #     print(day2addr2num,file=f)
    day2num = {}
    with open('fig4.txt','r',encoding='utf-8') as f:
        day2addr2num = literal_eval(f.read())
    for day,addr2num in day2addr2num.items():
        numlist = list(addr2num.values())
        maxnum = max(numlist)
        day2num[day] = maxnum
    # print(day2num)
    x = []
    y = []
    for day,num in day2num.items():
        x.append(day)
        y.append(num)
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, s=1)
    plt.xticks(rotation=30)
    x_major_locator = MultipleLocator(100)
    ax = plt.gca()
    ax.set_ylabel('# of transactions')
    ax.set_xlabel('Date')
    ax.set_ylim(1)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.yscale('log')
    plt.savefig('fig4.jpg',bbox_inches = 'tight')
    plt.show()
    return
    # with open('fig4.txt','r',encoding='utf-8') as f:
    #     day2addr2num = literal_eval(f.read())
    #将月份和交易数量存入x轴和y轴
    x = []
    y = []
    for mon,addr2num in day2addr2num.items():
        for addr,num in addr2num.items():
            x.append(mon)
            y.append(num)
    plt.scatter(x, y,s=1)
    plt.xticks(rotation=30)
    x_major_locator = MultipleLocator(90)
    ax = plt.gca()
    ax.set_ylabel('# of transactions')
    ax.set_xlabel('Date')
    ax.set_ylim(1)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.yscale('log')
    plt.savefig('fig4.jpg')
    plt.show()
def fig43intxnum():#fig4.4.1
    # with open('normal2athirdIntxnum.txt', 'r') as f:
    #     normal2Intxnum = literal_eval(f.read())
    # with open('scamInTx.txt','r') as f:
    #     scamInTxnum = literal_eval(f.read())
    # intx2nornum = {}
    # intx2scamnum = {}
    # for addr,txnum in normal2Intxnum.items():
    #     intx2nornum[txnum] = intx2nornum.get(txnum,0) + 1
    # for addr,txnum in scamInTxnum.items():
    #     intx2scamnum[txnum] = intx2scamnum.get(txnum,0) + 1
    # with open('intx2nornum.txt', 'w') as f:
    #     print(intx2nornum,file=f)
    # with open('intx2scamnum.txt', 'w') as f:
    #     print(intx2scamnum,file=f)

    with open('intx2nornum.txt', 'r') as f:
        intx2nornum = literal_eval(f.read())
    with open('intx2scamnum.txt', 'r') as f:
        intx2scamnum = literal_eval(f.read())
    x = ['num<10', '10≤num<20', '20≤num<50', '50≤num<200', '200≤num<500', '500≤num<1000',
         '1000≤num<2000', 'num>2000']
    y1 = [0, 0, 0, 0, 0, 0, 0, 0]
    y2 = [0, 0, 0, 0, 0, 0, 0, 0]
    index = np.arange(len(x))
    width = 0.4
    for intxnum,noraddrnum in intx2nornum.items():
        if intxnum < 10:
            y1[0] += intx2nornum[intxnum]
        if 10 <= intxnum and intxnum < 20:
            y1[1] += intx2nornum[intxnum]
        if 20 <= intxnum and intxnum < 50:
            y1[2] += intx2nornum[intxnum]
        if 50 <= intxnum and intxnum < 200:
            y1[3] += intx2nornum[intxnum]
        if 200 <= intxnum and intxnum < 500:
            y1[4] += intx2nornum[intxnum]
        if 500 <= intxnum and intxnum < 1000:
            y1[5] += intx2nornum[intxnum]
        if 1000 <= intxnum and intxnum < 2000:
            y1[6] += intx2nornum[intxnum]
        if 2000 <= intxnum:
            y1[7] += intx2nornum[intxnum]
    for intxnum,noraddrnum in intx2scamnum.items():
        if intxnum < 10:
            y2[0] += intx2scamnum[intxnum]
        if 10 <= intxnum and intxnum < 20:
            y2[1] += intx2scamnum[intxnum]
        if 20 <= intxnum and intxnum < 50:
            y2[2] += intx2scamnum[intxnum]
        if 50 <= intxnum and intxnum < 200:
            y2[3] += intx2scamnum[intxnum]
        if 200 <= intxnum and intxnum < 500:
            y2[4] += intx2scamnum[intxnum]
        if 500 <= intxnum and intxnum < 1000:
            y2[5] += intx2scamnum[intxnum]
        if 1000 <= intxnum and intxnum < 2000:
            y2[6] += intx2scamnum[intxnum]
        if 2000 <= intxnum:
            y2[7] += intx2scamnum[intxnum]
    plt.figure(figsize=(6, 5))
    plt.bar(x, y1, width, label="normal address in tx num",color='0.3')
    plt.bar(index + width, y2, width, label="scam address in tx num",color='0.9')

    for a, b in zip(x, y1):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)
    x = index + width
    for a, b in zip(x, y2):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)
    x = index + width * 2

    plt.xticks(rotation=30)
    ax = plt.gca()
    plt.yscale('log')
    ax.set_xlabel('in txs of normal/scam address')
    ax.set_ylabel('# of addr')
    plt.legend(loc='best')
    plt.savefig('fig43intx.jpg', bbox_inches='tight')
    plt.show()
def fig43outtxnum():
    # with open('normal2athirdOuttxnum.txt', 'r') as f:
    #     normal2Outtxnum = literal_eval(f.read())
    # with open('scamOutTx.txt','r') as f:
    #     scamOutTxnum = literal_eval(f.read())
    # outtx2nornum = {}
    # outtx2scamnum = {}
    # for addr,txnum in normal2Outtxnum.items():
    #     outtx2nornum[txnum] = outtx2nornum.get(txnum,0) + 1
    # for addr,txnum in scamOutTxnum.items():
    #     outtx2scamnum[txnum] = outtx2scamnum.get(txnum,0) + 1
    # with open('outtx2nornum.txt', 'w') as f:
    #     print(outtx2nornum,file=f)
    # with open('outtx2scamnum.txt', 'w') as f:
    #     print(outtx2scamnum,file=f)
    with open('outtx2nornum.txt', 'r') as f:
        outtx2nornum = literal_eval(f.read())
    with open('outtx2scamnum.txt', 'r') as f:
        outtx2scamnum = literal_eval(f.read())
    x = ['num<10', '10≤num<20', '20≤num<50', '50≤num<200', '200≤num<500', '500≤num<1000',
         '1000≤num<2000', 'num>2000']
    y1 = [0, 0, 0, 0, 0, 0, 0, 0]
    y2 = [0, 0, 0, 0, 0, 0, 0, 0]
    index = np.arange(len(x))
    width = 0.4
    for intxnum,noraddrnum in outtx2nornum.items():
        if intxnum < 10:
            y1[0] += outtx2nornum[intxnum]
        if 10 <= intxnum and intxnum < 20:
            y1[1] += outtx2nornum[intxnum]
        if 20 <= intxnum and intxnum < 50:
            y1[2] += outtx2nornum[intxnum]
        if 50 <= intxnum and intxnum < 200:
            y1[3] += outtx2nornum[intxnum]
        if 200 <= intxnum and intxnum < 500:
            y1[4] += outtx2nornum[intxnum]
        if 500 <= intxnum and intxnum < 1000:
            y1[5] += outtx2nornum[intxnum]
        if 1000 <= intxnum and intxnum < 2000:
            y1[6] += outtx2nornum[intxnum]
        if 2000 <= intxnum:
            y1[7] += outtx2nornum[intxnum]
    for intxnum,noraddrnum in outtx2scamnum.items():
        if intxnum < 10:
            y2[0] += outtx2scamnum[intxnum]
        if 10 <= intxnum and intxnum < 20:
            y2[1] += outtx2scamnum[intxnum]
        if 20 <= intxnum and intxnum < 50:
            y2[2] += outtx2scamnum[intxnum]
        if 50 <= intxnum and intxnum < 200:
            y2[3] += outtx2scamnum[intxnum]
        if 200 <= intxnum and intxnum < 500:
            y2[4] += outtx2scamnum[intxnum]
        if 500 <= intxnum and intxnum < 1000:
            y2[5] += outtx2scamnum[intxnum]
        if 1000 <= intxnum and intxnum < 2000:
            y2[6] += outtx2scamnum[intxnum]
        if 2000 <= intxnum:
            y2[7] += outtx2scamnum[intxnum]
    plt.figure(figsize=(6, 5))
    plt.bar(x, y1, width, label="normal address out tx num",color='0.3')
    plt.bar(index + width, y2, width, label="scam address out tx num",color='0.9')

    for a, b in zip(x, y1):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)
    x = index + width
    for a, b in zip(x, y2):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)
    x = index + width * 2
    plt.xticks(rotation=30)
    ax = plt.gca()
    plt.yscale('log')
    ax.set_xlabel('out txs of normal/scam address')
    ax.set_ylabel('# of addr')
    plt.legend(loc='best')
    plt.savefig('fig43outtx.jpg', bbox_inches='tight')
    plt.show()


def nfig6():#欺诈地址一般交易的living小时 scamNtxliving
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    # df1 = pandas.read_csv('ntx1.csv')
    # df2 = pandas.read_csv('ntx2.csv')
    # df3 = pandas.read_csv('ntx3.csv')
    # df4 = pandas.read_csv('ntx4.csv')
    # df5 = pandas.read_csv('ntx5.csv')
    # df6 = pandas.read_csv('bntx1.csv')
    # frames = [df1, df2, df3, df4, df5, df6]
    # df = pandas.concat(frames)
    # df = df.drop_duplicates()  # 去重
    df = pandas.read_csv('ntx.csv')
    df = df.sort_values('timeStamp')
    addr2time = {}#地址到交易时间戳的字典
    addr2living = {}
    for addr in addrlist:
        addr2time[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            addr2time[row['to']].append(row['timeStamp'])
    for addr,time in addr2time.items():#将时间列表排序，计算时间差
        time.sort()
    x = ['<6h','6h≤time<12h','12h≤time<18h','18h≤time<24h','24h≤time<48h','48h≤time<1 week','1 week≤time<1 month','>1 month']
    y = [0,0,0,0,0,0,0,0]
    for addr, time in addr2time.items():
        if len(time) > 0:
            start = datetime.datetime.fromtimestamp(time[0])
            end = datetime.datetime.fromtimestamp(time[-1])
            addr2living[addr] = (end - start).days * 24 + (end - start).seconds / 3600
    with open('scamNtxliving.txt','w') as f:
        print(addr2living,file=f)
    return
    for addr,living in addr2living.items():
        if living < 6:
            y[0] += 1
        if 6 <= living and living <12:
            y[1] += 1
        if 12 <= living and living < 18:
            y[2] += 1
        if 18 <= living and living < 24:
            y[3] += 1
        if 24 <= living and living < 48:
            y[4] += 1
        if 48 <= living and living < 168:
            y[5] += 1
        if 168 <= living and living < 720:
            y[6] += 1
        if 720 <= living:
            y[7] += 1
    return x,y
    print(y)
    plt.bar(x, y)
    plt.xticks(rotation=30)
    ax = plt.gca()
    plt.yscale('log')
    plt.show()

def ifig6():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    # df1 = pandas.read_csv('itx1.csv')
    # df2 = pandas.read_csv('itx2.csv')
    # df3 = pandas.read_csv('itx3.csv')
    # df4 = pandas.read_csv('itx4.csv')
    # df5 = pandas.read_csv('bitx1.csv')
    # frames = [df1, df2, df3, df4, df5]
    # df = pandas.concat(frames)
    df = pandas.read_csv('itx.csv')
    df = df.drop_duplicates()  # 去重
    df = df.sort_values('timeStamp')
    addr2time = {}#地址到交易时间戳的字典
    addr2living = {}
    for addr in addrlist:
        addr2time[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            addr2time[row['to']].append(row['timeStamp'])
    for addr,time in addr2time.items():#将时间列表排序，计算时间差
        time.sort()
    x = ['<6h','6h≤time<12h','12h≤time<18h','18h≤time<24h','24h≤time<48h','48h≤time<1 week','1 week≤time<1 month','>1 month']
    y = [0,0,0,0,0,0,0,0]
    for addr, time in addr2time.items():
        if len(time) > 0:
            start = datetime.datetime.fromtimestamp(time[0])
            end = datetime.datetime.fromtimestamp(time[-1])
            addr2living[addr] = (end - start).days * 24 + (end - start).seconds / 3600
    with open('scamItxliving.txt','w') as f:
        print(addr2living,file=f)
    for addr,living in addr2living.items():
        if living < 6:
            y[0] += 1
        if 6 <= living and living <12:
            y[1] += 1
        if 12 <= living and living < 18:
            y[2] += 1
        if 18 <= living and living < 24:
            y[3] += 1
        if 24 <= living and living < 48:
            y[4] += 1
        if 48 <= living and living < 168:
            y[5] += 1
        if 168 <= living and living < 720:
            y[6] += 1
        if 720 <= living:
            y[7] += 1
    return x,y
    print(y)
    plt.bar(x, y)
    plt.xticks(rotation=30)
    ax = plt.gca()
    plt.yscale('log')
    plt.show()

def efig6():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    # df1 = pandas.read_csv('etx1.csv')
    # df2 = pandas.read_csv('etx2.csv')
    # df3 = pandas.read_csv('etx3.csv')
    # df4 = pandas.read_csv('etx4.csv')
    # df5 = pandas.read_csv('betx1.csv')
    # frames = [df1, df2, df3, df4, df5]
    # df = pandas.concat(frames)
    # df = df.drop_duplicates()  # 去重
    df = pandas.read_csv('etx.csv')
    df = df.sort_values('timeStamp')
    addr2time = {}#地址到交易时间戳的字典
    addr2living = {}
    for addr in addrlist:
        addr2time[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            addr2time[row['to']].append(row['timeStamp'])
    for addr,time in addr2time.items():#将时间列表排序，计算时间差
        time.sort()
    # print(addr2time)
    x = ['<6h','6h≤time<12h','12h≤time<18h','18h≤time<24h','24h≤time<48h','48h≤time<1 week','1 week≤time<1 month','>1 month']
    y = [0,0,0,0,0,0,0,0]
    for addr, time in addr2time.items():
        if len(time) > 0:
            start = datetime.datetime.fromtimestamp(time[0])
            end = datetime.datetime.fromtimestamp(time[-1])
            addr2living[addr] = (end - start).days * 24 + (end - start).seconds / 3600
    with open('scamEtxliving.txt','w') as f:
        print(addr2living,file=f)
    for addr,living in addr2living.items():
        if living < 6:
            y[0] += 1
        if 6 <= living and living <12:
            y[1] += 1
        if 12 <= living and living < 18:
            y[2] += 1
        if 18 <= living and living < 24:
            y[3] += 1
        if 24 <= living and living < 48:
            y[4] += 1
        if 48 <= living and living < 168:
            y[5] += 1
        if 168 <= living and living < 720:
            y[6] += 1
        if 720 <= living:
            y[7] += 1
    return x,y
    print(y)
    plt.bar(x, y)
    plt.xticks(rotation=30)
    ax = plt.gca()
    plt.yscale('log')
    plt.show()

def fig6():
    nx,ny = nfig6()
    ny = numpy.array(ny)
    ix,iy = ifig6()
    iy = numpy.array(iy)
    ex,ey = efig6()
    ey = numpy.array(ey)
    y = numpy.array([1,1,1,1,1,1,1,1])
    x = ['<6h','6h≤time<12h','12h≤time<18h','18h≤time<24h','24h≤time<48h','48h≤time<1 week','1 week≤time<1 month','>1 month']
    x = numpy.arange(len(x))
    width = 0.2
    plt.bar(nx, ny, width, color='r', label="normal txs")
    plt.bar(x+width, iy, width, color='g', label="internal txs")
    plt.bar(x+width+width, ey, width, color='b', label="erc20 txs")
    plt.xticks(rotation=30)
    plt.yscale('log')
    plt.legend()
    plt.show()
def scamProfitRank():
    with open('scamNtxProfit.txt','r') as f:
        scamNtxProfit = literal_eval(f.read())
    with open('scamItxProfit.txt','r') as f:
        scamItxProfit = literal_eval(f.read())
    Profit2addr = {}
    # print(scamNtxProfit)
    zipped1 = zip(scamNtxProfit.keys(),scamNtxProfit.values())
    scamNtxProfit = sorted(zipped1,key = lambda x:(x[1]))
    zipped2 = zip(scamItxProfit.keys(),scamItxProfit.values())
    scamItxProfit = sorted(zipped2,key = lambda x:(x[1]))
    print(scamNtxProfit[-6:-1])
    print(scamItxProfit[-6:-1])
    #[('0x84f4bcc4e841afdfde7899662f8bbd7d6a71c112', 154030905.16457036),
    # ('0x4d53dad4b3552710ac276fa7f7ba944554c1cace', 247690286.23993436),和最后一个有关系
    # ('0x9409d7ea71294fc5561e5fdd47d4a9f5993948c5', 252262331.62993762),和下面一个的有关系
    # ('0x9a207194cbed9f229694fdf5a28caab59157920d', 275511997.60007125),
    # ('0x3408edca2d47ddaa783a3563d991b8ddebcd973b', 275512040.07925004)]

def fig7n():#pdf图
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('ntx.csv')
    # df = df.drop_duplicates()  # 去重
    df = df.sort_values('timeStamp')
    addr2prof = {}
    # url = "https://api.coingecko.com/api/v3/exchange_rates"
    # session = requests.Session()
    # res = literal_eval(session.get(url).text)['rates']
    # usd = res['usd']['value']  # 一个比特币的价格
    # rates = {}
    # for name, dic in res.items():
    #     rates[name] = usd / dic['value']
    # eth = rates['eth']  # 一个以太币的价格
    # eth = 2460.268204521556#欺诈地址之间互相转账不能计算
    # for index, row in df.iterrows():
    #     if isinstance(row['to'],str) and row['to'].lower() in addrlist and row['from'].lower() not in addrlist:
    #         addr2prof[row['to']] = 0
    # for index, row in df.iterrows():
    #     if isinstance(row['to'],str) and row['to'].lower() in addrlist and row['from'].lower() not in addrlist:
    #         try:
    #             if isinstance(row['value'],str):
    #                 addr2prof[row['to']] += int(row['value']) / 1000000000000000000 * eth
    #         except Exception:
    #             print(row['value'])
    # # print(addr2prof)
    # # with open('scamNtxProfit.txt','w') as f:
    # #     print(addr2prof,file=f)
    # # print(addr2prof['0x8909cc8d294544ca2c956550edbcb59ce4f2a9ad'])
    # # return
    # prof2num = {}
    profaxis = ['0-100k','100k-200k','200k-300k','300k-400k','400k-500k','500k-600k','600k-700k','700k-800k','800k-900k','900k-1000k','1000k-1100k','1100k-1200k','1200k-1300k','1300k-1400k','1400k-1500k','>1500k']
    # for prof in profaxis:
    #     prof2num[prof] = 0
    # for addr,prof in addr2prof.items():
    #     if prof / 1000 <= 100:
    #         prof2num['0-100k'] += 1
    #     if prof / 1000 > 100 and prof / 1000 <= 200:
    #         prof2num['100k-200k'] += 1
    #     if prof / 1000 > 200 and prof / 1000 <= 300:
    #         prof2num['200k-300k'] += 1
    #     if prof / 1000 > 300 and prof / 1000 <= 400:
    #         prof2num['300k-400k'] += 1
    #     if prof / 1000 > 400 and prof / 1000 <= 500:
    #         prof2num['400k-500k'] += 1
    #     if prof / 1000 > 500 and prof / 1000 <= 600:
    #         prof2num['500k-600k'] += 1
    #     if prof / 1000 > 600 and prof / 1000 <= 700:
    #         prof2num['600k-700k'] += 1
    #     if prof / 1000 > 700 and prof / 1000 <= 800:
    #         prof2num['700k-800k'] += 1
    #     if prof / 1000 > 800 and prof / 1000 <= 900:
    #         prof2num['800k-900k'] += 1
    #     if prof / 1000 > 900 and prof / 1000 <= 1000:
    #         prof2num['900k-1000k'] += 1
    #     if prof / 1000 > 1000 and prof / 1000 <= 1100:
    #         prof2num['1000k-1100k'] += 1
    #     if prof / 1000 > 1100 and prof / 1000 <= 1200:
    #         prof2num['1100k-1200k'] += 1
    #     if prof / 1000 > 1200 and prof / 1000 <= 1300:
    #         prof2num['1200k-1300k'] += 1
    #     if prof / 1000 > 1300 and prof / 1000 < 1400:
    #         prof2num['1300k-1400k'] += 1
    #     if prof / 1000 > 1400 and prof / 1000 < 1500:
    #         prof2num['1400k-1500k'] += 1
    #     if prof / 1000 > 1500:
    #         prof2num['>1500k'] += 1
    #         print(addr)
    # prof2num2 = {}
    # for addr,prof in addr2prof.items():
    #     addr2prof[addr] = addr2prof[addr]
    # for addr,prof in addr2prof.items():
    #     prof2num2[prof] = prof2num2.get(prof,0) + 1
    x = []
    y = []
    with open('fig7n.txt','r') as f:
        prof2num = literal_eval(f.read())
    # print(3257/sum(prof2num.values()))
    return
    # fig, ax = plt.subplots()
    for prof in profaxis:
        x.append(prof)
        y.append(prof2num[prof])
    plt.figure(figsize=(6.5,4.85))
    plt.bar(x,y,width=1)
    plt.xticks(rotation=30)
    plt.yscale('log')
    x_major_locator = MultipleLocator(5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.set_xlabel('The Profit of Address($)')
    ax.set_ylabel('the PDF graph of normal txs')
    plt.savefig('fig7n.jpg',bbox_inches = 'tight')
    plt.show()

    # with open('fig7n.txt','w') as f:
    #     print(prof2num,file=f)
    # with open('fig7n2.txt','w') as f:
    #     print(prof2num2,file=f)
    # with open('fig7n3.txt','w') as f:
    #     print(addr2prof,file=f)
def fig7nCdf():
    # with open('fig7n.txt','r') as f:
    #     prof2num = literal_eval(f.read())
    # # return
    # profaxis = ['0-100k','100k-200k','200k-300k','300k-400k','400k-500k','500k-600k','600k-700k','700k-800k','800k-900k','900k-1000k','1000k-1100k','1100k-1200k','1200k-1300k','1300k-1400k','1400k-1500k','>1500k']
    # x = []
    # y = []
    #
    # fig, ax = plt.subplots()
    # for prof in profaxis:
    #     x.append(prof)
    #     y.append(prof2num[prof])
    # plt.bar(x,y,width=1)
    # plt.xticks(rotation=30)
    # plt.yscale('log')
    # x_major_locator = MultipleLocator(5)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.savefig('fig7n.jpg')
    # ax.set_xlabel('The Profit of Address($)')
    # ax.set_ylabel('# of normal txs')
    # plt.show()
    prof2num = {}
    with open('fig7n2.txt','r') as f:
        prof2num = literal_eval(f.read())
    with open('fig7n3.txt','r') as f:
        addr2prof = literal_eval(f.read())
    proflist = sorted(prof2num.keys())
    print(proflist)
    print(max(proflist))
    for addr,prof in addr2prof.items():
        if prof == max(proflist):
            print(addr)
    # return
    maxp = max(proflist)
    # testprofit = []
    # for profit in proflist:
    #     if profit > 1000000:
    #         testprofit.append(profit)
    for addr,prof in addr2prof.items():
        if prof == maxp:
            print(addr)
    zipped = zip(prof2num.keys(), prof2num.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    fig, ax = plt.subplots()
    # for prof in profaxis:
    #     x.append(prof)
    #     y.append(prof2num[prof])
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x,percentage)
    plt.xscale('log')
    # x_major_locator = MultipleLocator(1000)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    ax.set_ylabel('CDF of Addresses')
    ax.set_xlabel('the profit of address($)')
    ax.set_xlim(1)
    # plt.savefig('fig7nCdf2.jpg')#x轴取log前
    plt.savefig('fig7nCdf2.jpg')#x轴取log后
    plt.show()

def fig7i():#pdf图
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('itx.csv')
    df = df.sort_values('timeStamp')
    addr2prof = {}
    # url = "https://api.coingecko.com/api/v3/exchange_rates"
    # session = requests.Session()
    # res = literal_eval(session.get(url).text)['rates']
    # usd = res['usd']['value']  # 一个比特币的价格
    # rates = {}
    # for name, dic in res.items():
    #     rates[name] = usd / dic['value']
    # eth = rates['eth']  # 一个以太币的价格
    eth = 2460.268204521556
    for index, row in df.iterrows():
        # if isinstance(row['to'],str):
        # if not math.isnan(row['to']):
        if isinstance(row['to'],str):
            addr2prof[row['to']] = 0
    # print(addr2prof)
    hashlist = []
    valuesum = 0
    for index, row in df.iterrows():
        # if isinstance(row['to'],str) and row['to'].lower() in addrlist and row['from'].lower() not in addrlist:
        try:
            if isinstance(row['to'],str) and row['to'].lower() in addrlist and row['from'].lower() not in addrlist:
                addr2prof[row['to']] += int(row['value']) / 1000000000000000000 * eth
            if isinstance(row['to'],str) and row['to'].lower() == '0xa9bf70a420d364e923c74448d9d817d3f2a77822' and int(row['value']) != 0 and row['from'] not in addrlist:
                # print(row)
                # print(row['value'])
                # print(row['hash'])
                hashlist.append(row['hash'])
                valuesum += int(row['value'])
        except Exception:
            print(row['value'])
    # print(addr2prof)
    # with open('scamItxProfit.txt','w') as f:
    #     print(addr2prof,file=f)
    # print(hashlist)
    # print(valuesum)
    #'0xa6074142f75502f24a050e0ed6d45a303d713051d56433f007e661d18d045900'是0xa9bf70a420d364e923c74448d9d817d3f2a77822欺诈最多的交易hash
    prof2num = {}
    profaxis = ['0-100k', '100k-200k', '200k-300k', '300k-400k', '400k-500k', '500k-600k', '600k-700k', '700k-800k',
                '800k-900k', '900k-1000k', '1000k-1100k', '1100k-1200k', '1200k-1300k', '1300k-1400k', '1400k-1500k',
                '>1500k']
    for prof in profaxis:
        prof2num[prof] = 0
    for addr, prof in addr2prof.items():
        if prof / 1000 <= 100:
            prof2num['0-100k'] += 1
        if prof / 1000 > 100 and prof / 1000 <= 200:
            prof2num['100k-200k'] += 1
        if prof / 1000 > 200 and prof / 1000 <= 300:
            prof2num['200k-300k'] += 1
        if prof / 1000 > 300 and prof / 1000 <= 400:
            prof2num['300k-400k'] += 1
        if prof / 1000 > 400 and prof / 1000 <= 500:
            prof2num['400k-500k'] += 1
        if prof / 1000 > 500 and prof / 1000 <= 600:
            prof2num['500k-600k'] += 1
        if prof / 1000 > 600 and prof / 1000 <= 700:
            prof2num['600k-700k'] += 1
        if prof / 1000 > 700 and prof / 1000 <= 800:
            prof2num['700k-800k'] += 1
        if prof / 1000 > 800 and prof / 1000 <= 900:
            prof2num['800k-900k'] += 1
        if prof / 1000 > 900 and prof / 1000 <= 1000:
            prof2num['900k-1000k'] += 1
        if prof / 1000 > 1000 and prof / 1000 <= 1100:
            prof2num['1000k-1100k'] += 1
        if prof / 1000 > 1100 and prof / 1000 <= 1200:
            prof2num['1100k-1200k'] += 1
        if prof / 1000 > 1200 and prof / 1000 <= 1300:
            prof2num['1200k-1300k'] += 1
        if prof / 1000 > 1300 and prof / 1000 < 1400:
            prof2num['1300k-1400k'] += 1
        if prof / 1000 > 1400 and prof / 1000 < 1500:
            prof2num['1400k-1500k'] += 1
        if prof / 1000 > 1500:
            prof2num['>1500k'] += 1
    prof2num2 = {}
    for addr, prof in addr2prof.items():
        addr2prof[addr] = int(addr2prof[addr])
    for addr, prof in addr2prof.items():
        prof2num2[prof] = prof2num2.get(prof, 0) + 1
    x = []
    y = []
    for prof in profaxis:
        x.append(prof)
        y.append(prof2num[prof])
    plt.figure(figsize=(6,6.5))
    plt.bar(x,y,width=1)
    plt.xticks(rotation=30)
    plt.yscale('log')
    x_major_locator = MultipleLocator(5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.set_ylabel('the PDF graph of internal txs')
    ax.set_xlabel('the profit of address($)')
    plt.savefig('fig7i.jpg',bbox_inches = 'tight')
    # plt.show()
    # with open('fig7i.txt','w') as f:
    #     print(prof2num,file=f)
    # with open('fig7i2.txt','w') as f:
    #     print(prof2num2,file=f)
    # with open('fig7i3.txt','w') as f:
    #     print(addr2prof,file=f)
def fig7iCdf():
    with open('fig7i2.txt', 'r') as f:
        prof2num = literal_eval(f.read())
    with open('fig7i3.txt', 'r') as f:
        addr2prof = literal_eval(f.read())
    # profaxis = ['0-100k','100k-200k','200k-300k','300k-400k','400k-500k']
    # ,'500k-600k','600k-700k','700k-800k','800k-900k','900k-1000k','1000k-1100k','1100k-1200k','1200k-1300k','1300k-1400k','1400k-1500k','>1500k']
    proflist = sorted(prof2num.keys())
    print(proflist)
    # print(max(proflist))
    maxp = max(proflist)
    # testprofit = []
    # for profit in proflist:
    #     if profit > 1000000:
    #         testprofit.append(profit)
    for addr, prof in addr2prof.items():
        if prof == maxp:
            print(addr)
    zipped = zip(prof2num.keys(), prof2num.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    fig, ax = plt.subplots()
    # for prof in profaxis:
    #     x.append(prof)
    #     y.append(prof2num[prof])
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x, percentage)
    plt.xscale('log')
    # x_major_locator = MultipleLocator(1000)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    ax.set_ylabel('CDF of Addresses')
    ax.set_xlabel('the profit of address($)')
    ax.set_xlim(1)
    plt.savefig('fig7iCdf2.jpg')
    # ax.set_xlim(1)
    plt.show()
def fig8():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    url = "https://api.coingecko.com/api/v3/exchange_rates"
    session = requests.Session()
    res = literal_eval(session.get(url).text)['rates']
    usd = res['usd']['value']#一个比特币的价格
    rates = {}
    for name,dic in res.items():
        rates[name] = usd / dic['value']
    eth = rates['eth']#一个以太币的价格
    print(res)
    print(usd)
    print(rates)
    print(eth)

    df = pandas.read_csv('etx.csv')
    addr2profit = {}
    for index, row in df.iterrows():
        try:
            if isinstance(row['to'],str) and row['to'] in addrlist:
                if isinstance(row['tokenSymbol'],str) and row['tokenSymbol'].lower() in rates.keys():
                    addr2profit[row['to']] = int(row['value']) * rates[row['tokenSymbol'].lower()] / 1000000000000000000
                elif not isinstance(row['tokenSymbol'],str) and not math.isnan((row['tokenSymbol'])):
                    continue
        except Exception:
            # print(index)
            print(math.isnan((row['tokenSymbol'])))
            print(row)
            print(type(row['tokenSymbol']))
            print(numpy.isnan(row['tokenSymbol']))
            # print(row['tokenSymbol'])
    print(addr2profit)
    print(len(addr2profit.keys()))
    return
    df5 = pandas.read_csv('itx1.csv')
    df6 = pandas.read_csv('itx2.csv')
    df7 = pandas.read_csv('itx3.csv')
    df8 = pandas.read_csv('itx4.csv')
    frames = [df5, df6, df7, df8]
    df = pandas.concat(frames)
    for index, row in df.iterrows():
        pass
def nfig9Intx():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('ntx.csv')
    #欺诈地址作为接收方的欺诈交易数量
    addr2num = {}
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            try:
                if isinstance(row['to'], str):
                    addr2num[row['to']] = addr2num.get(row['to'],0) + 1
            except Exception:
                print(row['to'])
    with open('scamInNtx.txt','w',encoding='utf-8') as f:
        print(addr2num,file=f)
    return
    # print(addr2num)
    # txnum2addrnum = {'1-10': 3447, '10-100': 1650, '100-1000': 228, '1000-5000': 24}
    # #先画直方图，横轴是交易数量区间，纵轴是欺诈地址数
    txnum2addrnum = {}
    # #原本要分的区间，不需要分区间，需要交易数量到地址数量的映射，如果该交易数量没有则构造
    # print(list(txnum2addrnum.values()))
    for addr,num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
    # print(txnum2addrnum)
    zipped = zip(txnum2addrnum.keys(), txnum2addrnum.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    with open('scamAddrNFig9Intx.txt','w',encoding='utf-8') as f:
        print(txnum2addrnum,file=f)
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x,percentage)
    plt.xscale('log')
    plt.show()

def fig921():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist21 = literal_eval(f.read())
    with open('scamInTx.txt','r') as f:
        addr2Intxnum = literal_eval(f.read())
    with open('scamOutTx.txt','r') as f:
        addr2Outtxnum = literal_eval(f.read())
    addr2Intxnum21 = {}
    addr2Outtxnum21 = {}
    for addr in addrlist21:#不是所有21年的欺诈地址都有in tx
        if addr in addr2Intxnum.keys():
            addr2Intxnum21[addr] = addr2Intxnum[addr]
        if addr in addr2Outtxnum.keys():
            addr2Outtxnum21[addr] = addr2Outtxnum[addr]
    with open('scamInTx21.txt','w') as f:
        print(addr2Intxnum21,file=f)
    with open('scamOutTx21.txt','w') as f:
        print(addr2Outtxnum21,file=f)
def ifig9Intx():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('itx.csv')
    addr2num = {}
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            try:
                if isinstance(row['to'], str):
                    addr2num[row['to']] = addr2num.get(row['to'],0) + 1
            except Exception:
                print(row['to'])
    with open('scamInItx.txt','w',encoding='utf-8') as f:
        print(addr2num,file=f)
    return
    txnum2addrnum = {}
    for addr,num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
    zipped = zip(txnum2addrnum.keys(), txnum2addrnum.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    with open('scamAddrIFig9Intx.txt','w',encoding='utf-8') as f:
        print(txnum2addrnum,file=f)
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x, percentage)
    plt.xscale('log')
    plt.show()
def efig9Intx():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('etx.csv')
    addr2num = {}
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            try:
                if isinstance(row['to'], str):
                    addr2num[row['to']] = addr2num.get(row['to'],0) + 1
            except Exception:
                print(row['to'])
    txnum2addrnum = {}
    for addr,num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
    zipped = zip(txnum2addrnum.keys(), txnum2addrnum.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    with open('scamAddrEFig9Intx.txt','w',encoding='utf-8') as f:
        print(txnum2addrnum,file=f)
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x, percentage)
    plt.xscale('log')
    plt.show()
def nfig9Outtx():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('ntx.csv')
    addr2num = {}
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:
            try:
                if isinstance(row['from'], str):
                    addr2num[row['from']] = addr2num.get(row['from'],0) + 1
            except Exception:
                print(row['from'])
    with open('scamOutNtx.txt','w',encoding='utf-8') as f:
        print(addr2num,file=f)
    return
    txnum2addrnum = {}
    for addr,num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
    zipped = zip(txnum2addrnum.keys(), txnum2addrnum.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    with open('scamAddrNFig9Outtx.txt','w',encoding='utf-8') as f:
        print(txnum2addrnum,file=f)
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x, percentage)
    plt.xscale('log')
    plt.show()
def ifig9Outtx():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    # df1 = pandas.read_csv('itx1.csv')
    # df2 = pandas.read_csv('itx2.csv')
    # df3 = pandas.read_csv('itx3.csv')
    # df4 = pandas.read_csv('itx4.csv')
    # df5 = pandas.read_csv('bitx1.csv')
    # frames = [df1, df2, df3, df4, df5]
    # df = pandas.concat(frames)
    df = pandas.read_csv('itx.csv')

    addr2num = {}
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:
            try:
                if isinstance(row['from'], str):
                    addr2num[row['from']] = addr2num.get(row['from'],0) + 1
            except Exception:
                print(row['from'])
    with open('scamOutItx.txt','w',encoding='utf-8') as f:
        print(addr2num,file=f)
    return
    txnum2addrnum = {}
    for addr,num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
    zipped = zip(txnum2addrnum.keys(), txnum2addrnum.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    with open('scamAddrIFig9Outtx.txt','w',encoding='utf-8') as f:
        print(txnum2addrnum,file=f)
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x, percentage)
    plt.xscale('log')
    plt.show()
def efig9Outtx():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    # df1 = pandas.read_csv('etx1.csv')
    # df2 = pandas.read_csv('etx2.csv')
    # df3 = pandas.read_csv('etx3.csv')
    # df4 = pandas.read_csv('etx4.csv')
    # df5 = pandas.read_csv('betx1.csv')
    # frames = [df1, df2, df3, df4, df5]
    # df = pandas.concat(frames)
    df = pandas.read_csv('etx.csv')
    addr2num = {}
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:
            try:
                if isinstance(row['from'], str):
                    addr2num[row['from']] = addr2num.get(row['from'],0) + 1
            except Exception:
                print(row['from'])
    txnum2addrnum = {}
    for addr,num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
    zipped = zip(txnum2addrnum.keys(), txnum2addrnum.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    with open('scamAddrEFig9Outtx.txt','w',encoding='utf-8') as f:
        print(txnum2addrnum,file=f)
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x, percentage)
    plt.xscale('log')
    plt.show()

# def tofig9():
#     with open('addr.txt','r',encoding='utf-8') as f:
#         addrlist = literal_eval(f.read())
#     df1 = pandas.read_csv('ntx1.csv')
#     df2 = pandas.read_csv('ntx2.csv')
#     df3 = pandas.read_csv('ntx3.csv')
#     df4 = pandas.read_csv('ntx4.csv')
#     df5 = pandas.read_csv('itx1.csv')
#     df6 = pandas.read_csv('itx2.csv')
#     df7 = pandas.read_csv('itx3.csv')
#     df8 = pandas.read_csv('itx4.csv')
#     df9 = pandas.read_csv('bntx1.csv')
#     df10 = pandas.read_csv('bitx1.csv')
#     frames = [df1, df2, df3, df4,df5,df6,df7,df8,df9,df10]
#     df = pandas.concat(frames)
#     addr2num = {}
#     for index, row in df.iterrows():
#         if isinstance(row['to'],str) and row['to'] in addrlist:
#             try:
#                 if isinstance(row['to'], str):
#                     addr2num[row['to']] = addr2num.get(row['to'],0) + 1
#             except Exception:
#                 print(row['to'])
#     txnum2addrnum = {}
#     for addr,num in addr2num.items():
#         txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
#     # print(txnum2addrnum)
#     zipped = zip(txnum2addrnum.keys(), txnum2addrnum.values())
#     sort_zipped = sorted(zipped, key=lambda x: (x[0]))
#     result = zip(*sort_zipped)
#     x, y = [list(x) for x in result]
#     with open('scamAddrToFig9.txt','w',encoding='utf-8') as f:
#         print(txnum2addrnum,file=f)
#     fig, ax = plt.subplots()
#     cum = numpy.cumsum(y)
#     percentage = cum / list(cum)[-1]
#     line = ax.plot(x, percentage)
#     plt.xscale('log')
#     plt.show()
# def fromfig9():
#     with open('addr.txt','r',encoding='utf-8') as f:
#         addrlist = literal_eval(f.read())
#     df1 = pandas.read_csv('ntx1.csv')
#     df2 = pandas.read_csv('ntx2.csv')
#     df3 = pandas.read_csv('ntx3.csv')
#     df4 = pandas.read_csv('ntx4.csv')
#     df5 = pandas.read_csv('itx1.csv')
#     df6 = pandas.read_csv('itx2.csv')
#     df7 = pandas.read_csv('itx3.csv')
#     df8 = pandas.read_csv('itx4.csv')
#     df9 = pandas.read_csv('bntx1.csv')
#     df10 = pandas.read_csv('bitx1.csv')
#     frames = [df1, df2, df3, df4,df5,df6,df7,df8,df9,df10]
#     df = pandas.concat(frames)
#     addr2num = {}
#     for index, row in df.iterrows():
#         if isinstance(row['to'],str) and row['from'] in addrlist:
#             try:
#                 if isinstance(row['from'], str):
#                     addr2num[row['from']] = addr2num.get(row['from'],0) + 1
#             except Exception:
#                 print(row['from'])
#     txnum2addrnum = {}
#     for addr,num in addr2num.items():
#         txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
#     zipped = zip(txnum2addrnum.keys(), txnum2addrnum.values())
#     sort_zipped = sorted(zipped, key=lambda x: (x[0]))
#     result = zip(*sort_zipped)
#     x, y = [list(x) for x in result]
#     with open('scamAddrFromFig9.txt','w',encoding='utf-8') as f:
#         print(txnum2addrnum,file=f)
#     fig, ax = plt.subplots()
#     cum = numpy.cumsum(y)
#     percentage = cum / list(cum)[-1]
#     line = ax.plot(x, percentage)
#     plt.xscale('log')
#     plt.show()
def normalAddrtx2csv1():#一般交易的from，to，value列
    df1 = pandas.read_csv('normalntx1.csv', low_memory=False, usecols=[1, 2, 6, 7, 8, 11, 14])#hash,from,to,value,contractAddress
    df2 = pandas.read_csv('normalntx2.csv', low_memory=False, usecols=[1, 2, 6, 7, 8, 11, 14])
    df3 = pandas.read_csv('normalntx3.csv', low_memory=False, usecols=[1, 2, 6, 7, 8, 11, 14])
    df4 = pandas.read_csv('normalntx4.csv', low_memory=False, usecols=[1, 2, 6, 7, 8, 11, 14])
    df5 = pandas.read_csv('normalntx5.csv', low_memory=False, usecols=[1, 2, 6, 7, 8, 11, 14])
    df6 = pandas.read_csv('normalntx6.csv', low_memory=False, usecols=[1, 2, 6, 7, 8, 11, 14])
    df7 = pandas.read_csv('normalntx7.csv', low_memory=False, usecols=[1, 2, 6, 7, 8, 11, 14])
    df8 = pandas.read_csv('normalntx8.csv', low_memory=False, usecols=[1, 2, 6, 7, 8, 11, 14])
    df9 = pandas.read_csv('normalntx9.csv', low_memory=False, usecols=[1, 2, 6, 7, 8, 11, 14])
    df10 = pandas.read_csv('normalntx10.csv', low_memory=False, usecols=[1, 2, 6, 7, 8, 11, 14])
    df11 = pandas.read_csv('normalntx11.csv', low_memory=False, usecols=[1, 2, 6, 7, 8, 11, 14])
    df12 = pandas.read_csv('normalntx12.csv', low_memory=False, usecols=[1, 2, 6, 7, 8, 11, 14])
    df13 = pandas.read_csv('normalntx13.csv', low_memory=False, usecols=[1, 2, 6, 7, 8, 11, 14])
    frames = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13]
    df = pandas.concat(frames)
    df.to_csv('normalAddrntx.csv')
def normalAddrtx2csv2():#内部交易的from，to，value列
    df14 = pandas.read_csv('normalitx1.csv', low_memory=False, usecols=[1, 2, 3, 4, 5, 6, 12])#hash,from,to,value
    df15 = pandas.read_csv('normalitx2.csv', low_memory=False, usecols=[1, 2, 3, 4, 5, 6, 12])
    df16 = pandas.read_csv('normalitx3.csv', low_memory=False, usecols=[1, 2, 3, 4, 5, 6, 12])
    df17 = pandas.read_csv('normalitx4.csv', low_memory=False, usecols=[1, 2, 3, 4, 5, 6, 12])
    df18 = pandas.read_csv('normalitx5.csv', low_memory=False, usecols=[1, 2, 3, 4, 5, 6, 12])
    df19 = pandas.read_csv('normalitx6.csv', low_memory=False, usecols=[1, 2, 3, 4, 5, 6, 12])
    df20 = pandas.read_csv('normalitx7.csv', low_memory=False, usecols=[1, 2, 3, 4, 5, 6, 12])
    df21 = pandas.read_csv('normalitx8.csv', low_memory=False, usecols=[1, 2, 3, 4, 5, 6, 12])
    df22 = pandas.read_csv('normalitx9.csv', low_memory=False, usecols=[1, 2, 3, 4, 5, 6, 12])
    df23 = pandas.read_csv('normalitx10.csv', low_memory=False, usecols=[1, 2, 3, 4, 5, 6, 12])
    frames = [df14, df15, df16, df17, df18, df19, df20, df21, df22, df23]
    df = pandas.concat(frames)
    df.to_csv('normalAddritx.csv')
def normalAddrtx2csv3():#内部交易的from，to，value列
    df14 = pandas.read_csv('normaletx1.csv', low_memory=False, usecols=[1, 2, 5, 7, 8, 10])#hash,from,to,value,tokenSymbol
    df15 = pandas.read_csv('normaletx2.csv', low_memory=False, usecols=[1, 2, 5, 7, 8, 10])
    df16 = pandas.read_csv('normaletx3.csv', low_memory=False, usecols=[1, 2, 5, 7, 8, 10])
    df17 = pandas.read_csv('normaletx4.csv', low_memory=False, usecols=[1, 2, 5, 7, 8, 10])
    df18 = pandas.read_csv('normaletx5.csv', low_memory=False, usecols=[1, 2, 5, 7, 8, 10])
    df19 = pandas.read_csv('normaletx6.csv', low_memory=False, usecols=[1, 2, 5, 7, 8, 10])
    df20 = pandas.read_csv('normaletx7.csv', low_memory=False, usecols=[1, 2, 5, 7, 8, 10])
    df21 = pandas.read_csv('normaletx8.csv', low_memory=False, usecols=[1, 2, 5, 7, 8, 10])
    df22 = pandas.read_csv('normaletx9.csv', low_memory=False, usecols=[1, 2, 5, 7, 8, 10])
    frames = [df14, df15, df16, df17, df18, df19, df20, df21, df22]
    df = pandas.concat(frames)
    df.to_csv('normalAddretx.csv')
def norAddrNFig9Intxs1():#统计normalAddrntx.csv里面的In Txs of normal Addresses（一般交易）
    with open('normalAddr.txt','r',encoding='utf-8') as f:#正常地址的一般交易的输入交易数量
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv',nrows=3000000)
    #欺诈地址作为接收方的欺诈交易数量
    addr2num = {}
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            try:
                if isinstance(row['to'], str):
                    addr2num[row['to']] = addr2num.get(row['to'],0) + 1
            except Exception:
                print(row['to'])
        print(index)
    txnum2addrnum = {}
    for addr,num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
    with open('norAddrNFig9Intxs1.txt','w',encoding='utf-8') as f:
        print(txnum2addrnum,file=f)
    return
    with open('nor2numNIntx1.txt','w',encoding='utf-8') as f:
        print(addr2num,file=f)


    # 统计交易数量和地址数量的映射
    # #先画直方图，横轴是交易数量区间，纵轴是欺诈地址数
    # txnum2addrnum = {}
    # # txnum = ['1-10','10-100','100-1000','1000-5000']#原本要分的区间，不需要分区间，需要交易数量到地址数量的映射，如果该交易数量没有则构造
    # # print(list(txnum2addrnum.values()))
    # for addr,num in addr2num.items():
    #     txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
    # zipped = zip(txnum2addrnum.keys(), txnum2addrnum.values())
    # sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    # result = zip(*sort_zipped)
    # x, y = [list(x) for x in result]
    # with open('norAddrNFig9Intxs.txt','w',encoding='utf-8') as f:
    #     print(txnum2addrnum,file=f)
    # fig, ax = plt.subplots()
    # cum = numpy.cumsum(y)
    # percentage = cum / list(cum)[-1]
    # line = ax.plot(x,percentage)
    # plt.xscale('log')
    # plt.show()
def norAddrNFig9Intxs2():#统计normalAddrntx.csv里面的In Txs of normal Addresses（一般交易）
    # with open('normalAddr.txt','r',encoding='utf-8') as f:
    #     addrlist = literal_eval(f.read())
    # df = pandas.read_csv('normalAddrntx2.csv').iloc[3000000:6000000]
    # #欺诈地址作为接收方的欺诈交易数量
    # addr2num = {}
    # for index, row in df.iterrows():
    #     if isinstance(row['to'],str) and row['to'] in addrlist:
    #         try:
    #             if isinstance(row['to'], str):
    #                 addr2num[row['to']] = addr2num.get(row['to'],0) + 1
    #         except Exception:
    #             print(row['to'])
    #     print(index)
    with open('nor2numNIntx2.txt','r',encoding='utf-8') as f:
        addr2num = literal_eval(f.read())
    txnum2addrnum = {}
    for addr,num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
    with open('norAddrNFig9Intxs2.txt','w',encoding='utf-8') as f:
        print(txnum2addrnum,file=f)
    return
    with open('nor2numNIntx2.txt','w',encoding='utf-8') as f:
        print(addr2num,file=f)
    return
    # 统计交易数量和地址数量的映射
    # #先画直方图，横轴是交易数量区间，纵轴是欺诈地址数
    txnum2addrnum = {}
    # txnum = ['1-10','10-100','100-1000','1000-5000']#原本要分的区间，不需要分区间，需要交易数量到地址数量的映射，如果该交易数量没有则构造
    # print(list(txnum2addrnum.values()))
    for addr,num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
    zipped = zip(txnum2addrnum.keys(), txnum2addrnum.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    with open('norAddrNFig9Intxs.txt','w',encoding='utf-8') as f:
        print(txnum2addrnum,file=f)
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x,percentage)
    plt.xscale('log')
    plt.show()
def norAddrNFig9Intxs3():#统计normalAddrntx.csv里面的In Txs of normal Addresses（一般交易）
    # with open('normalAddr.txt','r',encoding='utf-8') as f:
    #     addrlist = literal_eval(f.read())
    # df = pandas.read_csv('normalAddrntx2.csv')
    # df = df.iloc[6000000:9000000]
    # #欺诈地址作为接收方的欺诈交易数量
    # addr2num = {}
    # for index, row in df.iterrows():
    #     if isinstance(row['to'],str) and row['to'] in addrlist:
    #         try:
    #             if isinstance(row['to'], str):
    #                 addr2num[row['to']] = addr2num.get(row['to'],0) + 1
    #         except Exception:
    #             print(row['to'])
    #     print(index)
    with open('nor2numNIntx3.txt', 'r', encoding='utf-8') as f:
        addr2num = literal_eval(f.read())
    txnum2addrnum = {}
    for addr,num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
    with open('norAddrNFig9Intxs3.txt','w',encoding='utf-8') as f:
        print(txnum2addrnum,file=f)
    return
    with open('nor2numNIntx3.txt','w',encoding='utf-8') as f:
        print(addr2num,file=f)
    return
    # 统计交易数量和地址数量的映射
    # #先画直方图，横轴是交易数量区间，纵轴是欺诈地址数
    txnum2addrnum = {}
    # txnum = ['1-10','10-100','100-1000','1000-5000']#原本要分的区间，不需要分区间，需要交易数量到地址数量的映射，如果该交易数量没有则构造
    # print(list(txnum2addrnum.values()))
    for addr,num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
    zipped = zip(txnum2addrnum.keys(), txnum2addrnum.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    with open('norAddrNFig9Intxs.txt','w',encoding='utf-8') as f:
        print(txnum2addrnum,file=f)
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x,percentage)
    plt.xscale('log')
    plt.show()
def norAddrNFig9Intxs4():#统计normalAddrntx.csv里面的In Txs of normal Addresses（一般交易）
    # with open('normalAddr.txt','r',encoding='utf-8') as f:
    #     addrlist = literal_eval(f.read())
    # df = pandas.read_csv('normalAddrntx2.csv')
    # df = df.iloc[9000000:12000000]
    # #欺诈地址作为接收方的欺诈交易数量
    # addr2num = {}
    # for index, row in df.iterrows():
    #     if isinstance(row['to'],str) and row['to'] in addrlist:
    #         try:
    #             if isinstance(row['to'], str):
    #                 addr2num[row['to']] = addr2num.get(row['to'],0) + 1
    #         except Exception:
    #             print(row['to'])
    #     # print(index)
    with open('nor2numNIntx4.txt', 'r', encoding='utf-8') as f:
        addr2num = literal_eval(f.read())
    txnum2addrnum = {}
    for addr,num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
    with open('norAddrNFig9Intxs4.txt','w',encoding='utf-8') as f:
        print(txnum2addrnum,file=f)
    return

    with open('nor2numNIntx4.txt','w',encoding='utf-8') as f:
        print(addr2num,file=f)
    return
def norAddrNFig9Intxs5():#统计normalAddrntx.csv里面的In Txs of normal Addresses（一般交易）
    # with open('normalAddr.txt','r',encoding='utf-8') as f:
    #     addrlist = literal_eval(f.read())
    # df = pandas.read_csv('normalAddrntx2.csv')
    # df = df.iloc[12000000:,[0,4]]
    # #欺诈地址作为接收方的欺诈交易数量
    # addr2num = {}
    # for index, row in df.iterrows():
    #     if isinstance(row['to'],str) and row['to'] in addrlist:
    #         try:
    #             if isinstance(row['to'], str):
    #                 addr2num[row['to']] = addr2num.get(row['to'],0) + 1
    #         except Exception:
    #             print(row['to'])
    #     print(index)

    with open('nor2numNIntx5.txt', 'r', encoding='utf-8') as f:
        addr2num = literal_eval(f.read())
    txnum2addrnum = {}
    for addr,num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
    with open('norAddrNFig9Intxs5.txt','w',encoding='utf-8') as f:
        print(txnum2addrnum,file=f)
    return
    with open('nor2numNIntx5.txt','w',encoding='utf-8') as f:
        print(addr2num,file=f)
    return
#统计txnum2addrnum
def mixnorAddrNFig9Intxs():
    with open('norAddrNFig9Intxs1.txt', 'r', encoding='utf-8') as f:
        txnum2addrnum1 = literal_eval(f.read())
    with open('norAddrNFig9Intxs2.txt', 'r', encoding='utf-8') as f:
        txnum2addrnum2 = literal_eval(f.read())
    with open('norAddrNFig9Intxs3.txt', 'r', encoding='utf-8') as f:
        txnum2addrnum3 = literal_eval(f.read())
    with open('norAddrNFig9Intxs4.txt', 'r', encoding='utf-8') as f:
        txnum2addrnum4 = literal_eval(f.read())
    with open('norAddrNFig9Intxs5.txt', 'r', encoding='utf-8') as f:
        txnum2addrnum5 = literal_eval(f.read())
    txnum2addrnum = {}
    txnumlist = list(txnum2addrnum1.keys()) + list(txnum2addrnum2.keys()) + list(txnum2addrnum3.keys()) + list(txnum2addrnum4.keys()) + list(txnum2addrnum5.keys())
    for txnum in txnumlist:
        if txnum in txnum2addrnum1.keys():
            txnum2addrnum[txnum] = txnum2addrnum.get(txnum,0) + txnum2addrnum1[txnum]
        if txnum in txnum2addrnum2.keys():
            txnum2addrnum[txnum] = txnum2addrnum.get(txnum,0) + txnum2addrnum2[txnum]
        if txnum in txnum2addrnum3.keys():
            txnum2addrnum[txnum] = txnum2addrnum.get(txnum,0) + txnum2addrnum3[txnum]
        if txnum in txnum2addrnum4.keys():
            txnum2addrnum[txnum] = txnum2addrnum.get(txnum,0) + txnum2addrnum4[txnum]
        if txnum in txnum2addrnum5.keys():
            txnum2addrnum[txnum] = txnum2addrnum.get(txnum,0) + txnum2addrnum5[txnum]
    with open('norAddrNFig9Intxs.txt','w',encoding='utf-8') as f:
        print(txnum2addrnum,file=f)
def mixnorAddrNFig9Outtxs():
    with open('norAddrNFig9Outtxs1.txt', 'r', encoding='utf-8') as f:
        txnum2addrnum1 = literal_eval(f.read())
    with open('norAddrNFig9Outtxs2.txt', 'r', encoding='utf-8') as f:
        txnum2addrnum2 = literal_eval(f.read())
    with open('norAddrNFig9Outtxs3.txt', 'r', encoding='utf-8') as f:
        txnum2addrnum3 = literal_eval(f.read())
    with open('norAddrNFig9Outtxs4.txt', 'r', encoding='utf-8') as f:
        txnum2addrnum4 = literal_eval(f.read())
    with open('norAddrNFig9Outtxs5.txt', 'r', encoding='utf-8') as f:
        txnum2addrnum5 = literal_eval(f.read())
    txnum2addrnum = {}
    txnumlist = list(txnum2addrnum1.keys()) + list(txnum2addrnum2.keys()) + list(txnum2addrnum3.keys()) + list(txnum2addrnum4.keys()) + list(txnum2addrnum5.keys())
    for txnum in txnumlist:
        if txnum in txnum2addrnum1.keys():
            txnum2addrnum[txnum] = txnum2addrnum.get(txnum,0) + txnum2addrnum1[txnum]
        if txnum in txnum2addrnum2.keys():
            txnum2addrnum[txnum] = txnum2addrnum.get(txnum,0) + txnum2addrnum2[txnum]
        if txnum in txnum2addrnum3.keys():
            txnum2addrnum[txnum] = txnum2addrnum.get(txnum,0) + txnum2addrnum3[txnum]
        if txnum in txnum2addrnum4.keys():
            txnum2addrnum[txnum] = txnum2addrnum.get(txnum,0) + txnum2addrnum4[txnum]
        if txnum in txnum2addrnum5.keys():
            txnum2addrnum[txnum] = txnum2addrnum.get(txnum,0) + txnum2addrnum5[txnum]
    with open('norAddrNFig9Outtxs.txt','w',encoding='utf-8') as f:
        print(txnum2addrnum,file=f)
#internal tx不需要分段分析交易，正常地址的内部交易的输入交易数量

def norAddrIFig9Intxs():#统计normalAddritx.csv里面的In Txs of normal Addresses（内部交易）
    # with open('normalAddr.txt','r',encoding='utf-8') as f:
    #     addrlist = literal_eval(f.read())
    # df = pandas.read_csv('normalAddritx2.csv')
    # addr2num = {}
    # for index, row in df.iterrows():
    #     if isinstance(row['to'],str) and row['to'] in addrlist:
    #         try:
    #             if isinstance(row['to'], str):
    #                 addr2num[row['to']] = addr2num.get(row['to'],0) + 1
    #         except Exception:
    #             print(row['to'])
    #     print(index)
    txnum2addrnum = {}
    with open('nor2numIIntx1.txt', 'r', encoding='utf-8') as f:
        addr2num = literal_eval(f.read())
    for addr,num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
    with open('norAddrIFig9Intxs.txt','w',encoding='utf-8') as f:
        print(txnum2addrnum,file=f)
    return
    with open('nor2numIIntx1.txt','w',encoding='utf-8') as f:
        print(addr2num,file=f)
    return
    for addr,num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
    zipped = zip(txnum2addrnum.keys(), txnum2addrnum.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    with open('norAddrIFig9Intxs.txt','w',encoding='utf-8') as f:
        print(txnum2addrnum,file=f)
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x, percentage)
    plt.xscale('log')
    plt.show()
#token tx不需要分段
def norAddrEFig9Intxs():#正常地址的token交易的输入交易数量
    # with open('normalAddr.txt','r',encoding='utf-8') as f:
    #     addrlist = literal_eval(f.read())
    # df = pandas.read_csv('normalAddretx2.csv')
    # #欺诈地址作为接收方的欺诈交易数量
    # addr2num = {}
    # for index, row in df.iterrows():
    #     if isinstance(row['to'],str) and row['to'] in addrlist:
    #         try:
    #             if isinstance(row['to'], str):
    #                 addr2num[row['to']] = addr2num.get(row['to'],0) + 1
    #         except Exception:
    #             print(row['to'])
    #     print(index)
    txnum2addrnum = {}
    with open('nor2numEIntx1.txt', 'r', encoding='utf-8') as f:
        addr2num = literal_eval(f.read())
    for addr,num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
    with open('norAddrEFig9Intxs.txt','w',encoding='utf-8') as f:
        print(txnum2addrnum,file=f)
    return
    with open('nor2numEIntx1.txt','w',encoding='utf-8') as f:
        print(addr2num,file=f)
    return

def norAddrNFig9Outtxs1():#统计两个文件的out txs
    # with open('normalAddr.txt','r',encoding='utf-8') as f:
    #     addrlist = literal_eval(f.read())
    # df = pandas.read_csv('normalAddrntx2.csv',nrows=3000000)#读取去重交易后的正常地址交易
    # df = df.iloc[0:,[0,3]]
    # pandas.set_option('display.max_columns', None)
    # addr2num = {}
    # for index, row in df.iterrows():
    #     if isinstance(row['from'],str) and row['from'] in addrlist:
    #         try:
    #             if isinstance(row['from'], str):
    #                 addr2num[row['from']] = addr2num.get(row['from'],0) + 1
    #         except Exception:
    #             print(row['from'])
    #     print(index)
    with open('nor2numNOuttx1.txt','r',encoding='utf-8') as f:
        addr2num = literal_eval(f.read())
    txnum2addrnum = {}
    for addr, num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num, 0) + 1
    with open('norAddrNFig9Outtxs1.txt', 'w', encoding='utf-8') as f:
        print(txnum2addrnum, file=f)
    return
    with open('nor2numNOuttx1.txt','w',encoding='utf-8') as f:
        print(addr2num,file=f)
    return
    # for addr,num in addr2num.items():
    #     txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
    # with open('normalAddrFromFig91.txt','w',encoding='utf-8') as f:
    #     print(txnum2addrnum,file=f)
    # zipped = zip(txnum2addrnum.keys(), txnum2addrnum.values())
    # sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    # result = zip(*sort_zipped)
    # x, y = [list(x) for x in result]
    # fig, ax = plt.subplots()
    # cum = numpy.cumsum(y)
    # percentage = cum / list(cum)[-1]
    # line = ax.plot(x, percentage)
    # plt.xscale('log')
    # plt.show()
def norAddrNFig9Outtxs2():#统计两个文件的out txs
    # with open('normalAddr.txt','r',encoding='utf-8') as f:
    #     addrlist = literal_eval(f.read())
    # df = pandas.read_csv('normalAddrntx2.csv',usecols=[0,3]).iloc[3000000:6000000]
    # pandas.set_option('display.max_columns', None)
    # addr2num = {}
    # for index, row in df.iterrows():
    #     if isinstance(row['from'],str) and row['from'] in addrlist:
    #         try:
    #             if isinstance(row['from'], str):
    #                 addr2num[row['from']] = addr2num.get(row['from'],0) + 1
    #         except Exception:
    #             print(row['from'])
    #     print(index)
    # txnum2addrnum = {}
    with open('nor2numNOuttx2.txt','r',encoding='utf-8') as f:
        addr2num = literal_eval(f.read())
    txnum2addrnum = {}
    for addr, num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num, 0) + 1
    with open('norAddrNFig9Outtxs2.txt', 'w', encoding='utf-8') as f:
        print(txnum2addrnum, file=f)
    return
    with open('nor2numNOuttx2.txt','w',encoding='utf-8') as f:
        print(addr2num,file=f)
    return
    for addr,num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
    with open('normalAddrFromFig92.txt','w',encoding='utf-8') as f:
        print(txnum2addrnum,file=f)
    zipped = zip(txnum2addrnum.keys(), txnum2addrnum.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x, percentage)
    plt.xscale('log')
    plt.show()
def norAddrNFig9Outtxs3():#统计两个文件的out txs
    # with open('normalAddr.txt','r',encoding='utf-8') as f:
    #     addrlist = literal_eval(f.read())
    # df = pandas.read_csv('normalAddrntx2.csv',usecols=[0,3]).iloc[6000000:9000000]
    # pandas.set_option('display.max_columns', None)
    # addr2num = {}
    # for index, row in df.iterrows():
    #     if isinstance(row['from'],str) and row['from'] in addrlist:
    #         try:
    #             if isinstance(row['from'], str):
    #                 addr2num[row['from']] = addr2num.get(row['from'],0) + 1
    #         except Exception:
    #             print(row['from'])
    #     print(index)
    # txnum2addrnum = {}
    with open('nor2numNOuttx3.txt','r',encoding='utf-8') as f:
        addr2num = literal_eval(f.read())
    txnum2addrnum = {}
    for addr, num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num, 0) + 1
    with open('norAddrNFig9Outtxs3.txt', 'w', encoding='utf-8') as f:
        print(txnum2addrnum, file=f)
    return
    with open('nor2numNOuttx3.txt','w',encoding='utf-8') as f:
        print(addr2num,file=f)
    return
    for addr,num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num,0) + 1
    with open('normalAddrFromFig92.txt','w',encoding='utf-8') as f:
        print(txnum2addrnum,file=f)
    zipped = zip(txnum2addrnum.keys(), txnum2addrnum.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x, percentage)
    plt.xscale('log')
    plt.show()
def norAddrNFig9Outtxs4():#统计两个文件的out txs
    # with open('normalAddr.txt','r',encoding='utf-8') as f:
    #     addrlist = literal_eval(f.read())
    # df = pandas.read_csv('normalAddrntx2.csv',usecols=[0,3]).iloc[9000000:12000000]
    # pandas.set_option('display.max_columns', None)
    # addr2num = {}
    # for index, row in df.iterrows():
    #     if isinstance(row['from'],str) and row['from'] in addrlist:
    #         try:
    #             if isinstance(row['from'], str):
    #                 addr2num[row['from']] = addr2num.get(row['from'],0) + 1
    #         except Exception:
    #             print(row['from'])
    #     print(index)
    # txnum2addrnum = {}
    with open('nor2numNOuttx4.txt','r',encoding='utf-8') as f:
        addr2num = literal_eval(f.read())
    txnum2addrnum = {}
    for addr, num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num, 0) + 1
    with open('norAddrNFig9Outtxs4.txt', 'w', encoding='utf-8') as f:
        print(txnum2addrnum, file=f)
    return
    with open('nor2numNOuttx4.txt','w',encoding='utf-8') as f:
        print(addr2num,file=f)
    return
def norAddrNFig9Outtxs5():#统计两个文件的out txs
    # with open('normalAddr.txt','r',encoding='utf-8') as f:
    #     addrlist = literal_eval(f.read())
    # df = pandas.read_csv('normalAddrntx2.csv',usecols=[0,3]).iloc[12000000:]
    # pandas.set_option('display.max_columns', None)
    # addr2num = {}
    # for index, row in df.iterrows():
    #     if isinstance(row['from'],str) and row['from'] in addrlist:
    #         try:
    #             if isinstance(row['from'], str):
    #                 addr2num[row['from']] = addr2num.get(row['from'],0) + 1
    #         except Exception:
    #             print(row['from'])
    #     print(index)
    # txnum2addrnum = {}
    with open('nor2numNOuttx5.txt','r',encoding='utf-8') as f:
        addr2num = literal_eval(f.read())
    txnum2addrnum = {}
    for addr, num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num, 0) + 1
    with open('norAddrNFig9Outtxs5.txt', 'w', encoding='utf-8') as f:
        print(txnum2addrnum, file=f)
    return
    with open('nor2numNOuttx5.txt','w',encoding='utf-8') as f:
        print(addr2num,file=f)
    return
def norAddrIFig9Outtxs():#统计两个文件的out txs
    # with open('normalAddr.txt','r',encoding='utf-8') as f:
    #     addrlist = literal_eval(f.read())
    # df = pandas.read_csv('normalAddritx2.csv',usecols=[0,3])
    # pandas.set_option('display.max_columns', None)
    # addr2num = {}
    # for index, row in df.iterrows():
    #     if isinstance(row['from'],str) and row['from'] in addrlist:
    #         try:
    #             if isinstance(row['from'], str):
    #                 addr2num[row['from']] = addr2num.get(row['from'],0) + 1
    #         except Exception:
    #             print(row['from'])
    #     print(index)
    with open('nor2numIOuttx.txt','r',encoding='utf-8') as f:
        addr2num = literal_eval(f.read())
    txnum2addrnum = {}
    for addr, num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num, 0) + 1
    with open('norAddrIFig9Outtxs.txt', 'w', encoding='utf-8') as f:
        print(txnum2addrnum, file=f)
    return
    with open('nor2numIOuttx.txt','w',encoding='utf-8') as f:
        print(addr2num,file=f)
    return
def norAddrEFig9Outtxs():#统计两个文件的out txs
    # with open('normalAddr.txt','r',encoding='utf-8') as f:
    #     addrlist = literal_eval(f.read())
    # df = pandas.read_csv('normalAddretx2.csv',usecols=[0,3])
    # pandas.set_option('display.max_columns', None)
    # addr2num = {}
    # for index, row in df.iterrows():
    #     if isinstance(row['from'],str) and row['from'] in addrlist:
    #         try:
    #             if isinstance(row['from'], str):
    #                 addr2num[row['from']] = addr2num.get(row['from'],0) + 1
    #         except Exception:
    #             print(row['from'])
    #     print(index)
    with open('nor2numEOuttx.txt','r',encoding='utf-8') as f:
        addr2num = literal_eval(f.read())
    txnum2addrnum = {}
    for addr, num in addr2num.items():
        txnum2addrnum[num] = txnum2addrnum.get(num, 0) + 1
    with open('norAddrEFig9Outtx.txt', 'w', encoding='utf-8') as f:
        print(txnum2addrnum, file=f)
    return
    # txnum2addrnum = {}
    with open('nor2numEOuttx.txt','w',encoding='utf-8') as f:
        print(addr2num,file=f)
    return

def getnormaladdr():
    df = pandas.read_csv('ethereum_tagged_address.csv',encoding='ISO-8859-1')
    normalAddr = []
    hacktype = ['blacklist']
    exchangenum = 0
    for index, row in df.iterrows():
        for typeword in hacktype:
            if typeword.lower() not in row['label'].lower():
                normalAddr.append(row['addr'])
            if 'exchange' in row['label'].lower():
                exchangenum += 1
    # print(exchangenum)
    normalAddr = list(set(normalAddr))
    # print(len(normalAddr))
    with open('Peckshield.txt','w') as f:
        print(normalAddr,file=f)
def getnormaladdr2():
    option = webdriver.ChromeOptions()
    option.add_argument('disable-infobars')
    with webdriver.Chrome() as w:
        w.get('https://cn.etherscan.com/login')
        username = w.find_element_by_xpath('//*[@id="ContentPlaceHolder1_txtUserName"]').send_keys('pegEth')
        passwd = w.find_element_by_xpath('//*[@id="ContentPlaceHolder1_txtPassword"]').send_keys('Asdljh0422')
        time.sleep(20)
        w.get('https://cn.etherscan.com/accounts/label/exchange?subcatid=undefined&size=10000&start=0&col=1&order=asc')
def getnormaladdrNtxs1():
    with open('etherscanLabeledExchange1.txt','r',encoding='utf-8') as f:
        list1 = literal_eval(f.read())
    print(len(list1))
    with open('etherscanLabeledExchange2.txt','r',encoding='utf-8') as f:
        list2 = literal_eval(f.read())
    print(len(list2))
    with open('Peckshield.txt','r',encoding='utf-8') as f:
        list3 = literal_eval(f.read())
    print(len(list3))
    mylist = list(set(list1 + list2 + list3))
    # with open('normalAddr.txt','w',encoding='utf-8') as f:
    #     print(mylist,file=f)
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    print(len(addrlist))
    return
    addrlist = addrlist[0:1500]
    session = requests.Session()
    with open('normalntx1.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","transactionIndex",
                         "from","to","value","gas","gasPrice","isError","txreceipt_status","input",
                         "contractAddress","cumulativeGasUsed","gasUsed","confirmations"])
        number = 0
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlist&address="+addr+"&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],result['blockHash'],
                                        result['transactionIndex'],result['from'],result['to'],result['value'],result['gas'],result['gasPrice'],
                                        result['isError'],result['txreceipt_status'],result['input'],result['contractAddress'],result['cumulativeGasUsed'],
                                        result['gasUsed'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrNtxs2():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[1500:3000]
    session = requests.Session()
    with open('normalntx2.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","transactionIndex",
                         "from","to","value","gas","gasPrice","isError","txreceipt_status","input",
                         "contractAddress","cumulativeGasUsed","gasUsed","confirmations"])
        number = 1500
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlist&address="+addr+"&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],result['blockHash'],
                                        result['transactionIndex'],result['from'],result['to'],result['value'],result['gas'],result['gasPrice'],
                                        result['isError'],result['txreceipt_status'],result['input'],result['contractAddress'],result['cumulativeGasUsed'],
                                        result['gasUsed'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrNtxs3():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[3000:4500]
    session = requests.Session()
    with open('normalntx3.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","transactionIndex",
                         "from","to","value","gas","gasPrice","isError","txreceipt_status","input",
                         "contractAddress","cumulativeGasUsed","gasUsed","confirmations"])
        number = 3000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlist&address="+addr+"&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],result['blockHash'],
                                        result['transactionIndex'],result['from'],result['to'],result['value'],result['gas'],result['gasPrice'],
                                        result['isError'],result['txreceipt_status'],result['input'],result['contractAddress'],result['cumulativeGasUsed'],
                                        result['gasUsed'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrNtxs4():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[4500:6000]
    session = requests.Session()
    with open('normalntx4.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","transactionIndex",
                         "from","to","value","gas","gasPrice","isError","txreceipt_status","input",
                         "contractAddress","cumulativeGasUsed","gasUsed","confirmations"])
        number = 4500
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlist&address="+addr+"&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],result['blockHash'],
                                        result['transactionIndex'],result['from'],result['to'],result['value'],result['gas'],result['gasPrice'],
                                        result['isError'],result['txreceipt_status'],result['input'],result['contractAddress'],result['cumulativeGasUsed'],
                                        result['gasUsed'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrNtxs5():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[6000:7500]
    session = requests.Session()
    with open('normalntx5.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","transactionIndex",
                         "from","to","value","gas","gasPrice","isError","txreceipt_status","input",
                         "contractAddress","cumulativeGasUsed","gasUsed","confirmations"])
        number = 6000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlist&address="+addr+"&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],result['blockHash'],
                                        result['transactionIndex'],result['from'],result['to'],result['value'],result['gas'],result['gasPrice'],
                                        result['isError'],result['txreceipt_status'],result['input'],result['contractAddress'],result['cumulativeGasUsed'],
                                        result['gasUsed'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrNtxs6():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[7500:9000]
    session = requests.Session()
    with open('normalntx6.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","transactionIndex",
                         "from","to","value","gas","gasPrice","isError","txreceipt_status","input",
                         "contractAddress","cumulativeGasUsed","gasUsed","confirmations"])
        number = 7500
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlist&address="+addr+"&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],result['blockHash'],
                                        result['transactionIndex'],result['from'],result['to'],result['value'],result['gas'],result['gasPrice'],
                                        result['isError'],result['txreceipt_status'],result['input'],result['contractAddress'],result['cumulativeGasUsed'],
                                        result['gasUsed'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrNtxs7():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[9000:10500]
    session = requests.Session()
    with open('normalntx7.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","transactionIndex",
                         "from","to","value","gas","gasPrice","isError","txreceipt_status","input",
                         "contractAddress","cumulativeGasUsed","gasUsed","confirmations"])
        number = 9000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlist&address="+addr+"&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],result['blockHash'],
                                        result['transactionIndex'],result['from'],result['to'],result['value'],result['gas'],result['gasPrice'],
                                        result['isError'],result['txreceipt_status'],result['input'],result['contractAddress'],result['cumulativeGasUsed'],
                                        result['gasUsed'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrNtxs8():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[10500:12000]
    session = requests.Session()
    with open('normalntx8.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","transactionIndex",
                         "from","to","value","gas","gasPrice","isError","txreceipt_status","input",
                         "contractAddress","cumulativeGasUsed","gasUsed","confirmations"])
        number = 10500
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlist&address="+addr+"&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],result['blockHash'],
                                        result['transactionIndex'],result['from'],result['to'],result['value'],result['gas'],result['gasPrice'],
                                        result['isError'],result['txreceipt_status'],result['input'],result['contractAddress'],result['cumulativeGasUsed'],
                                        result['gasUsed'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrNtxs9():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[12000:13500]
    session = requests.Session()
    with open('normalntx9.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","transactionIndex",
                         "from","to","value","gas","gasPrice","isError","txreceipt_status","input",
                         "contractAddress","cumulativeGasUsed","gasUsed","confirmations"])
        number = 12000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlist&address="+addr+"&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],result['blockHash'],
                                        result['transactionIndex'],result['from'],result['to'],result['value'],result['gas'],result['gasPrice'],
                                        result['isError'],result['txreceipt_status'],result['input'],result['contractAddress'],result['cumulativeGasUsed'],
                                        result['gasUsed'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrNtxs10():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[13500:15000]
    session = requests.Session()
    with open('normalntx10.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","transactionIndex",
                         "from","to","value","gas","gasPrice","isError","txreceipt_status","input",
                         "contractAddress","cumulativeGasUsed","gasUsed","confirmations"])
        number = 13500
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlist&address="+addr+"&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],result['blockHash'],
                                        result['transactionIndex'],result['from'],result['to'],result['value'],result['gas'],result['gasPrice'],
                                        result['isError'],result['txreceipt_status'],result['input'],result['contractAddress'],result['cumulativeGasUsed'],
                                        result['gasUsed'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrNtxs11():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[15000:16500]
    session = requests.Session()
    with open('normalntx11.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","transactionIndex",
                         "from","to","value","gas","gasPrice","isError","txreceipt_status","input",
                         "contractAddress","cumulativeGasUsed","gasUsed","confirmations"])
        number = 15000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlist&address="+addr+"&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],result['blockHash'],
                                        result['transactionIndex'],result['from'],result['to'],result['value'],result['gas'],result['gasPrice'],
                                        result['isError'],result['txreceipt_status'],result['input'],result['contractAddress'],result['cumulativeGasUsed'],
                                        result['gasUsed'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrNtxs12():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[16500:]
    session = requests.Session()
    with open('normalntx12.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","transactionIndex",
                         "from","to","value","gas","gasPrice","isError","txreceipt_status","input",
                         "contractAddress","cumulativeGasUsed","gasUsed","confirmations"])
        number = 16500
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlist&address="+addr+"&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],result['blockHash'],
                                        result['transactionIndex'],result['from'],result['to'],result['value'],result['gas'],result['gasPrice'],
                                        result['isError'],result['txreceipt_status'],result['input'],result['contractAddress'],result['cumulativeGasUsed'],
                                        result['gasUsed'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrNtxs13():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = ['0x7c49e1c0e33f3efb57d64b7690fa287c8d15b90a']
    session = requests.Session()
    with open('normalntx13.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","transactionIndex",
                         "from","to","value","gas","gasPrice","isError","txreceipt_status","input",
                         "contractAddress","cumulativeGasUsed","gasUsed","confirmations"])
        number = 16500
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlist&address="+addr+"&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],result['blockHash'],
                                        result['transactionIndex'],result['from'],result['to'],result['value'],result['gas'],result['gasPrice'],
                                        result['isError'],result['txreceipt_status'],result['input'],result['contractAddress'],result['cumulativeGasUsed'],
                                        result['gasUsed'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrItxs1():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[0:2000]
    session = requests.Session()
    with open('normalitx1.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","from","to","value",
                         "contractAddress","input","type","gas","gasUsed","traceId",
                         "isError","errCode"])
        number = 0
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlistinternal&address=" + addr + "&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            # print(url)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['from'],
                                         result['to'],result['value'],result['contractAddress'],result['input'],result['type'],
                                         result['gas'],result['gasUsed'],result['traceId'],result['isError'],result['errCode']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrItxs2():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[2000:4000]
    session = requests.Session()
    with open('normalitx2.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","from","to","value",
                         "contractAddress","input","type","gas","gasUsed","traceId",
                         "isError","errCode"])
        number = 2000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlistinternal&address=" + addr + "&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            # print(url)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['from'],
                                         result['to'],result['value'],result['contractAddress'],result['input'],result['type'],
                                         result['gas'],result['gasUsed'],result['traceId'],result['isError'],result['errCode']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrItxs3():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[4000:6000]
    session = requests.Session()
    with open('normalitx3.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","from","to","value",
                         "contractAddress","input","type","gas","gasUsed","traceId",
                         "isError","errCode"])
        number = 4000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlistinternal&address=" + addr + "&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            # print(url)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['from'],
                                         result['to'],result['value'],result['contractAddress'],result['input'],result['type'],
                                         result['gas'],result['gasUsed'],result['traceId'],result['isError'],result['errCode']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrItxs4():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[6000:8000]
    session = requests.Session()
    with open('normalitx4.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","from","to","value",
                         "contractAddress","input","type","gas","gasUsed","traceId",
                         "isError","errCode"])
        number = 6000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlistinternal&address=" + addr + "&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            # print(url)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['from'],
                                         result['to'],result['value'],result['contractAddress'],result['input'],result['type'],
                                         result['gas'],result['gasUsed'],result['traceId'],result['isError'],result['errCode']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrItxs5():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[8000:10000]
    session = requests.Session()
    with open('normalitx5.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","from","to","value",
                         "contractAddress","input","type","gas","gasUsed","traceId",
                         "isError","errCode"])
        number = 8000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlistinternal&address=" + addr + "&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            # print(url)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['from'],
                                         result['to'],result['value'],result['contractAddress'],result['input'],result['type'],
                                         result['gas'],result['gasUsed'],result['traceId'],result['isError'],result['errCode']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrItxs6():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[10000:12000]
    session = requests.Session()
    with open('normalitx6.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","from","to","value",
                         "contractAddress","input","type","gas","gasUsed","traceId",
                         "isError","errCode"])
        number = 10000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlistinternal&address=" + addr + "&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            # print(url)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['from'],
                                         result['to'],result['value'],result['contractAddress'],result['input'],result['type'],
                                         result['gas'],result['gasUsed'],result['traceId'],result['isError'],result['errCode']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrItxs7():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[12000:14000]
    session = requests.Session()
    with open('normalitx7.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","from","to","value",
                         "contractAddress","input","type","gas","gasUsed","traceId",
                         "isError","errCode"])
        number = 12000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlistinternal&address=" + addr + "&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            # print(url)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['from'],
                                         result['to'],result['value'],result['contractAddress'],result['input'],result['type'],
                                         result['gas'],result['gasUsed'],result['traceId'],result['isError'],result['errCode']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrItxs8():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[14000:16000]
    session = requests.Session()
    with open('normalitx8.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","from","to","value",
                         "contractAddress","input","type","gas","gasUsed","traceId",
                         "isError","errCode"])
        number = 14000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlistinternal&address=" + addr + "&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            # print(url)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['from'],
                                         result['to'],result['value'],result['contractAddress'],result['input'],result['type'],
                                         result['gas'],result['gasUsed'],result['traceId'],result['isError'],result['errCode']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrItxs9():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[16000:]
    session = requests.Session()
    with open('normalitx9.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","from","to","value",
                         "contractAddress","input","type","gas","gasUsed","traceId",
                         "isError","errCode"])
        number = 16000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlistinternal&address=" + addr + "&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            # print(url)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['from'],
                                         result['to'],result['value'],result['contractAddress'],result['input'],result['type'],
                                         result['gas'],result['gasUsed'],result['traceId'],result['isError'],result['errCode']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrItxs10():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = ['0x7c49e1c0e33f3efb57d64b7690fa287c8d15b90a']
    session = requests.Session()
    with open('normalitx10.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","from","to","value",
                         "contractAddress","input","type","gas","gasUsed","traceId",
                         "isError","errCode"])
        number = 16000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=txlistinternal&address=" + addr + "&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            # print(url)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['from'],
                                         result['to'],result['value'],result['contractAddress'],result['input'],result['type'],
                                         result['gas'],result['gasUsed'],result['traceId'],result['isError'],result['errCode']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrEtxs1():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[0:2000]
    session = requests.Session()
    with open('normaletx1.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","from",
                         "contractAddress","to","value","tokenName","tokenSymbol","tokenDecimal",
                         "transactionIndex","gas","gasPrice","gasUsed","cumulativeGasUsed","input","confirmations"])
        number = 0
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=tokentx&address=" + addr + "&page=1&offset=10000&startblock=0&endblock=99999999&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],
                                         result['blockHash'],result['from'],result['contractAddress'],result['to'],result['value'],
                                         result['tokenName'],result['tokenSymbol'],result['tokenDecimal'],result['transactionIndex'],result['gas'],
                                         result['gasPrice'],result['gasUsed'],result['cumulativeGasUsed'],result['input'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrEtxs2():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[2000:4000]
    session = requests.Session()
    with open('normaletx2.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","from",
                         "contractAddress","to","value","tokenName","tokenSymbol","tokenDecimal",
                         "transactionIndex","gas","gasPrice","gasUsed","cumulativeGasUsed","input","confirmations"])
        number = 2000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=tokentx&address=" + addr + "&page=1&offset=10000&startblock=0&endblock=99999999&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],
                                         result['blockHash'],result['from'],result['contractAddress'],result['to'],result['value'],
                                         result['tokenName'],result['tokenSymbol'],result['tokenDecimal'],result['transactionIndex'],result['gas'],
                                         result['gasPrice'],result['gasUsed'],result['cumulativeGasUsed'],result['input'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrEtxs3():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[4000:6000]
    session = requests.Session()
    with open('normaletx3.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","from",
                         "contractAddress","to","value","tokenName","tokenSymbol","tokenDecimal",
                         "transactionIndex","gas","gasPrice","gasUsed","cumulativeGasUsed","input","confirmations"])
        number = 4000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=tokentx&address=" + addr + "&page=1&offset=10000&startblock=0&endblock=99999999&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],
                                         result['blockHash'],result['from'],result['contractAddress'],result['to'],result['value'],
                                         result['tokenName'],result['tokenSymbol'],result['tokenDecimal'],result['transactionIndex'],result['gas'],
                                         result['gasPrice'],result['gasUsed'],result['cumulativeGasUsed'],result['input'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrEtxs4():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[6000:8000]
    session = requests.Session()
    with open('normaletx4.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","from",
                         "contractAddress","to","value","tokenName","tokenSymbol","tokenDecimal",
                         "transactionIndex","gas","gasPrice","gasUsed","cumulativeGasUsed","input","confirmations"])
        number = 6000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=tokentx&address=" + addr + "&page=1&offset=10000&startblock=0&endblock=99999999&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],
                                         result['blockHash'],result['from'],result['contractAddress'],result['to'],result['value'],
                                         result['tokenName'],result['tokenSymbol'],result['tokenDecimal'],result['transactionIndex'],result['gas'],
                                         result['gasPrice'],result['gasUsed'],result['cumulativeGasUsed'],result['input'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrEtxs5():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[8000:10000]
    session = requests.Session()
    with open('normaletx5.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","from",
                         "contractAddress","to","value","tokenName","tokenSymbol","tokenDecimal",
                         "transactionIndex","gas","gasPrice","gasUsed","cumulativeGasUsed","input","confirmations"])
        number = 8000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=tokentx&address=" + addr + "&page=1&offset=10000&startblock=0&endblock=99999999&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],
                                         result['blockHash'],result['from'],result['contractAddress'],result['to'],result['value'],
                                         result['tokenName'],result['tokenSymbol'],result['tokenDecimal'],result['transactionIndex'],result['gas'],
                                         result['gasPrice'],result['gasUsed'],result['cumulativeGasUsed'],result['input'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrEtxs6():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[10000:12000]
    session = requests.Session()
    with open('normaletx6.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","from",
                         "contractAddress","to","value","tokenName","tokenSymbol","tokenDecimal",
                         "transactionIndex","gas","gasPrice","gasUsed","cumulativeGasUsed","input","confirmations"])
        number = 10000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=tokentx&address=" + addr + "&page=1&offset=10000&startblock=0&endblock=99999999&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],
                                         result['blockHash'],result['from'],result['contractAddress'],result['to'],result['value'],
                                         result['tokenName'],result['tokenSymbol'],result['tokenDecimal'],result['transactionIndex'],result['gas'],
                                         result['gasPrice'],result['gasUsed'],result['cumulativeGasUsed'],result['input'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrEtxs7():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[12000:14000]
    session = requests.Session()
    with open('normaletx7.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","from",
                         "contractAddress","to","value","tokenName","tokenSymbol","tokenDecimal",
                         "transactionIndex","gas","gasPrice","gasUsed","cumulativeGasUsed","input","confirmations"])
        number = 12000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=tokentx&address=" + addr + "&page=1&offset=10000&startblock=0&endblock=99999999&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],
                                         result['blockHash'],result['from'],result['contractAddress'],result['to'],result['value'],
                                         result['tokenName'],result['tokenSymbol'],result['tokenDecimal'],result['transactionIndex'],result['gas'],
                                         result['gasPrice'],result['gasUsed'],result['cumulativeGasUsed'],result['input'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrEtxs8():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[14000:16000]
    session = requests.Session()
    with open('normaletx8.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","from",
                         "contractAddress","to","value","tokenName","tokenSymbol","tokenDecimal",
                         "transactionIndex","gas","gasPrice","gasUsed","cumulativeGasUsed","input","confirmations"])
        number = 14000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=tokentx&address=" + addr + "&page=1&offset=10000&startblock=0&endblock=99999999&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],
                                         result['blockHash'],result['from'],result['contractAddress'],result['to'],result['value'],
                                         result['tokenName'],result['tokenSymbol'],result['tokenDecimal'],result['transactionIndex'],result['gas'],
                                         result['gasPrice'],result['gasUsed'],result['cumulativeGasUsed'],result['input'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrEtxs9():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = addrlist[16000:]
    session = requests.Session()
    with open('normaletx9.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","from",
                         "contractAddress","to","value","tokenName","tokenSymbol","tokenDecimal",
                         "transactionIndex","gas","gasPrice","gasUsed","cumulativeGasUsed","input","confirmations"])
        number = 16000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=tokentx&address=" + addr + "&page=1&offset=10000&startblock=0&endblock=99999999&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],
                                         result['blockHash'],result['from'],result['contractAddress'],result['to'],result['value'],
                                         result['tokenName'],result['tokenSymbol'],result['tokenDecimal'],result['transactionIndex'],result['gas'],
                                         result['gasPrice'],result['gasUsed'],result['cumulativeGasUsed'],result['input'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def getnormaladdrEtxs10():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = ['0xc46ac314fa9ce95667031a26fffe05827af6f192','0x07c0201aa56fb130a4d6cd99e9ef7efb55fc6219','0x5e44d890f52432a01d467730672325184f521ae7']
    session = requests.Session()
    with open('normaletx9.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber","timeStamp","hash","nonce","blockHash","from",
                         "contractAddress","to","value","tokenName","tokenSymbol","tokenDecimal",
                         "transactionIndex","gas","gasPrice","gasUsed","cumulativeGasUsed","input","confirmations"])
        number = 16000
        for addr in addrlist:
            url = "https://api.etherscan.io/api?module=account&action=tokentx&address=" + addr + "&page=1&offset=10000&startblock=0&endblock=99999999&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            number += 1
            print(number)
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'],result['timeStamp'],result['hash'],result['nonce'],
                                         result['blockHash'],result['from'],result['contractAddress'],result['to'],result['value'],
                                         result['tokenName'],result['tokenSymbol'],result['tokenDecimal'],result['transactionIndex'],result['gas'],
                                         result['gasPrice'],result['gasUsed'],result['cumulativeGasUsed'],result['input'],result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def scamIncome():#欺诈地址的总收入
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df1 = pandas.read_csv('ntx.csv')
    df2 = pandas.read_csv('itx.csv')
    frames = [df1,df2]
    df = pandas.concat(frames)
    addr2income = {}
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            try:
                if isinstance(row['to'], str):
                    addr2income[row['to']] = addr2income.get(row['to'],0) + int(row['value']) / 1000000000000000000
            except Exception:
                print(row['to'])
                print(row['value'])
                traceback.print_exc()
    with open('scamIncome.txt','w',encoding='utf-8') as f:
        print(addr2income,file=f)

def scamOutcome():#欺诈地址的总支出
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df1 = pandas.read_csv('ntx.csv')
    df2 = pandas.read_csv('itx.csv')
    frames = [df1,df2]
    df = pandas.concat(frames)
    addr2outcome = {}
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:
            try:
                if isinstance(row['from'], str):
                    addr2outcome[row['from']] = addr2outcome.get(row['from'],0) + int(row['value']) / 1000000000000000000
            except Exception:
                print(row['scamOutcome'])
                print(row['value'])
                traceback.print_exc()
    with open('scamOutcome.txt','w',encoding='utf-8') as f:
        print(addr2outcome,file=f)
def scamTx():#欺诈地址in/out txnum
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df1 = pandas.read_csv('ntx.csv')
    df2 = pandas.read_csv('itx.csv')
    frames = [df1,df2]
    df = pandas.concat(frames)
    addr2Intxnum = {}
    addr2Outtxnum = {}
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            try:
                if isinstance(row['to'], str):
                    addr2Intxnum[row['to']] = addr2Intxnum.get(row['to'],0) + 1
            except Exception:
                print(row['to'])
                print(row['value'])
                traceback.print_exc()
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:
            try:
                if isinstance(row['from'], str):
                    addr2Outtxnum[row['from']] = addr2Outtxnum.get(row['from'],0) + 1
            except Exception:
                print(row['from'])
                traceback.print_exc()
    with open('scamInTx.txt','w') as f:#欺诈地址两种交易的in txs
        print(addr2Intxnum,file=f)
    with open('scamOutTx.txt','w') as f:
        print(addr2Outtxnum,file=f)

# def scamIOTxAna():
#     with open('scamInTx.txt','r') as f:
#         scamInTx = literal_eval(f.read())
#     with open('scamOutTx.txt','r') as f:
#         scamOutTx = literal_eval(f.read())
#     scamInTx = sorted(scamInTx.items(),key = lambda x:(x[1]))
#     scamOutTx = sorted(scamOutTx.items(),key = lambda x:(x[1]))
#     print(scamInTx)
#     print(scamOutTx)
    # with open('scamInNtx.txt','r') as f:
    #     scamInNtx = literal_eval(f.read())
    # with open('scamOutNtx.txt','r') as f:
    #     scamOutNtx = literal_eval(f.read())
    # scamInNtx = sorted(scamInNtx.items(),key = lambda x:(x[1]))
    # scamOutNtx = sorted(scamOutNtx.items(),key = lambda x:(x[1]))
    # with open('scamInItx.txt','r') as f:
    #     scamInItx = literal_eval(f.read())
    # with open('scamOutItx.txt','r') as f:
    #     scamOutItx = literal_eval(f.read())
    # scamInItx = sorted(scamInItx.items(),key = lambda x:(x[1]))
    # scamOutItx = sorted(scamOutItx.items(),key = lambda x:(x[1]))
    # print(scamInItx)
    # print(scamOutItx)
def scamIOTxnum2():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df1 = pandas.read_csv('ntx.csv')
    df2 = pandas.read_csv('itx.csv')
    frames = [df1,df2]
    df = pandas.concat(frames)
    addr2Intxnum = {}
    addr2Outtxnum = {}
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist and isinstance(row['from'],str) and row['from'] in addrlist:
            try:
                if isinstance(row['to'], str):
                    addr2Intxnum[row['to']] = addr2Intxnum.get(row['to'],0) + 1
            except Exception:
                print(row['to'])
                print(row['value'])
                traceback.print_exc()
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist and isinstance(row['from'],str) and row['from'] in addrlist:
            try:
                if isinstance(row['from'], str):
                    addr2Outtxnum[row['from']] = addr2Outtxnum.get(row['from'],0) + 1
            except Exception:
                print(row['from'])
                traceback.print_exc()
    with open('scamInTx2.txt','w') as f:#没用到这个文件，没用
        print(addr2Intxnum,file=f)
    with open('scamOutTx2.txt','w') as f:
        print(addr2Outtxnum,file=f)
def normalInNtx1():#正常地址normal tx的in tx num
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv',usecols=[0,4,5]).iloc[0:3000000]
    pandas.set_option('display.max_columns', None)
    addr2income = {}
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            try:
                if isinstance(row['to'], str):
                    addr2income[row['to']] = addr2income.get(row['to'], 0) + int(row['value']) / 1000000000000000000
            except Exception:
                print(row['to'])
                print(row['value'])
                traceback.print_exc()
        print(index)
    with open('normalIncomeNtx1.txt','w',encoding='utf-8') as f:
        print(addr2income,file=f)
def normalInNtx2():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv',usecols=[0,4,5]).iloc[3000000:6000000]
    pandas.set_option('display.max_columns', None)
    addr2income = {}
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            try:
                if isinstance(row['to'], str):
                    addr2income[row['to']] = addr2income.get(row['to'], 0) + int(row['value']) / 1000000000000000000
            except Exception:
                print(row['to'])
                print(row['value'])
                traceback.print_exc()
        print(index)
    with open('normalIncomeNtx2.txt','w',encoding='utf-8') as f:
        print(addr2income,file=f)
def normalInNtx3():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv',usecols=[0,4,5]).iloc[6000000:9000000]
    pandas.set_option('display.max_columns', None)
    addr2income = {}
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            try:
                if isinstance(row['to'], str):
                    addr2income[row['to']] = addr2income.get(row['to'], 0) + int(row['value']) / 1000000000000000000
            except Exception:
                print(row['to'])
                print(row['value'])
                traceback.print_exc()
        print(index)
    with open('normalIncomeNtx3.txt','w',encoding='utf-8') as f:
        print(addr2income,file=f)
def normalInNtx4():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv',usecols=[0,4,5]).iloc[9000000:12000000]
    pandas.set_option('display.max_columns', None)
    addr2income = {}
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            try:
                if isinstance(row['to'], str):
                    addr2income[row['to']] = addr2income.get(row['to'], 0) + int(row['value']) / 1000000000000000000
            except Exception:
                print(row['to'])
                print(row['value'])
                traceback.print_exc()
        print(index)
    with open('normalIncomeNtx4.txt','w',encoding='utf-8') as f:
        print(addr2income,file=f)
def normalInNtx5():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv',usecols=[0,4,5]).iloc[12000000:]
    pandas.set_option('display.max_columns', None)
    addr2income = {}
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            try:
                if isinstance(row['to'], str):
                    addr2income[row['to']] = addr2income.get(row['to'], 0) + int(row['value']) / 1000000000000000000
            except Exception:
                print(row['to'])
                print(row['value'])
                traceback.print_exc()
        print(index)
    with open('normalIncomeNtx5.txt','w',encoding='utf-8') as f:
        print(addr2income,file=f)
#internal tx 不需要分段
def normalInItx():#正常地址internal tx的in tx num
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddritx2.csv',usecols=[0,4,5])
    pandas.set_option('display.max_columns', None)
    addr2income = {}
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            try:
                if isinstance(row['to'], str):
                    addr2income[row['to']] = addr2income.get(row['to'], 0) + int(row['value']) / 1000000000000000000
            except Exception:
                print(row['to'])
                print(row['value'])
                traceback.print_exc()
        print(index)
    with open('normalIncomeItx1.txt','w',encoding='utf-8') as f:
        print(addr2income,file=f)
#token tx不需要计算利润

def normalOutNtx1():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv',usecols=[0,3,5]).iloc[0:3000000]
    addr2outcome = {}
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:
            try:
                if isinstance(row['from'], str):
                    addr2outcome[row['from']] = addr2outcome.get(row['from'], 0) + int(row['value']) / 1000000000000000000
            except Exception:
                print(row['from'])
                print(row['value'])
                traceback.print_exc()
        print(index)
    with open('normalOutcomeNtx1.txt', 'w', encoding='utf-8') as f:
        print(addr2outcome, file=f)
def normalAvgOutNtx2():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv',usecols=[0,3,5]).iloc[3000000:6000000]
    addr2outcome = {}
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:
            try:
                if isinstance(row['from'], str):
                    addr2outcome[row['from']] = addr2outcome.get(row['from'], 0) + int(row['value']) / 1000000000000000000
            except Exception:
                print(row['from'])
                print(row['value'])
                traceback.print_exc()
        print(index)
    with open('normalOutcomeNtx2.txt', 'w', encoding='utf-8') as f:
        print(addr2outcome, file=f)
def normalOutNtx3():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv',usecols=[0,3,5]).iloc[6000000:9000000]
    addr2outcome = {}
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:
            try:
                if isinstance(row['from'], str):
                    addr2outcome[row['from']] = addr2outcome.get(row['from'], 0) + int(row['value']) / 1000000000000000000
            except Exception:
                print(row['from'])
                print(row['value'])
                traceback.print_exc()
        print(index)
    with open('normalOutcomeNtx3.txt', 'w', encoding='utf-8') as f:
        print(addr2outcome, file=f)
def normalOutNtx4():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv',usecols=[0,3,5]).iloc[9000000:12000000]
    addr2outcome = {}
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:
            try:
                if isinstance(row['from'], str):
                    addr2outcome[row['from']] = addr2outcome.get(row['from'], 0) + int(row['value']) / 1000000000000000000
            except Exception:
                print(row['from'])
                print(row['value'])
                traceback.print_exc()
        print(index)
    with open('normalOutcomeNtx4.txt', 'w', encoding='utf-8') as f:
        print(addr2outcome, file=f)
def normalOutNtx5():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv',usecols=[0,3,5]).iloc[12000000:]
    addr2outcome = {}
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:
            try:
                if isinstance(row['from'], str):
                    addr2outcome[row['from']] = addr2outcome.get(row['from'], 0) + int(row['value']) / 1000000000000000000
            except Exception:
                print(row['from'])
                print(row['value'])
                traceback.print_exc()
        print(index)
    with open('normalOutcomeNtx5.txt', 'w', encoding='utf-8') as f:
        print(addr2outcome, file=f)
#internal tx不需要分段
def normalOutItx():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddritx2.csv',usecols=[0,3,5])
    addr2outcome = {}
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:
            try:
                if isinstance(row['from'], str):
                    addr2outcome[row['from']] = addr2outcome.get(row['from'], 0) + int(row['value']) / 1000000000000000000
            except Exception:
                print(row['from'])
                print(row['value'])
                traceback.print_exc()
        print(index)
    with open('normalOutcomeItx1.txt', 'w', encoding='utf-8') as f:
        print(addr2outcome, file=f)
#token tx不需要计算income outcome
def normalAvgOutEtx():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddretx2.csv',usecols=[0,3,5])
    addr2outcome = {}
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:
            try:
                if isinstance(row['from'], str):
                    addr2outcome[row['from']] = addr2outcome.get(row['from'], 0) + int(row['value']) / 1000000000000000000
            except Exception:
                print(row['from'])
                print(row['value'])
                traceback.print_exc()
        print(index)
    with open('normalAvgOutEtx1.txt', 'w', encoding='utf-8') as f:
        print(addr2outcome, file=f)

def mixScamAvg():#算成欺诈地址总收入了
    with open('scamIncome.txt','r',encoding='utf-8') as f:
        addr2income = literal_eval(f.read())
    zipped1 = zip(addr2income.keys(), addr2income.values())
    sort_zipped1 = sorted(zipped1, key=lambda x: (x[0]))
    result1 = zip(*sort_zipped1)
    x1, y1 = [list(x) for x in result1]
    cum = numpy.cumsum(y1)
    percentage1 = cum / list(cum)[-1]
    with open('scamOutcome.txt','r',encoding='utf-8') as f:
        addr2outcome = literal_eval(f.read())
    zipped2 = zip(addr2outcome.keys(), addr2outcome.values())
    sort_zipped2 = sorted(zipped2, key=lambda x: (x[0]))
    result2 = zip(*sort_zipped2)
    x2, y2 = [list(x) for x in result2]
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y2)
    percentage2 = cum / list(cum)[-1]
    line1 = ax.plot(x1, percentage1, label='Avg Income Of scam addresses', color='black')
    line2 = ax.plot(x2, percentage2, label='Avg Outcome Of scam addresses', linestyle="--", color='black')
    ax.set_xlabel('Average Income/Outcome of Addresses (ETH)')
    ax.set_ylabel('CDF of Addresses')
    plt.xscale('log')
    plt.legend()
    plt.show()
def mixNormalAvg():
    with open('normalAvgIn.txt', 'r', encoding='utf-8') as f:
        addr2income = literal_eval(f.read())
    zipped1 = zip(addr2income.keys(), addr2income.values())
    sort_zipped1 = sorted(zipped1, key=lambda x: (x[0]))
    result1 = zip(*sort_zipped1)
    x1, y1 = [list(x) for x in result1]
    cum = numpy.cumsum(y1)
    percentage1 = cum / list(cum)[-1]
    with open('normalAvgOut.txt','r',encoding='utf-8') as f:
        addr2outcome = literal_eval(f.read())
    zipped2 = zip(addr2outcome.keys(), addr2outcome.values())
    sort_zipped2 = sorted(zipped2, key=lambda x: (x[0]))
    result2 = zip(*sort_zipped2)
    x2, y2 = [list(x) for x in result2]
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y2)
    percentage2 = cum / list(cum)[-1]
    # line1 = ax.plot(x1, percentage1, label='Avg Income Of normal addresses', color='black')
    line2 = ax.plot(x2, percentage2, label='Avg Outcome Of normal addresses', linestyle="--", color='black')
    ax.set_xlabel('Average Income/Outcome of Addresses (ETH)')
    ax.set_ylabel('CDF of Addresses')
    plt.xscale('log')
    plt.legend()
    plt.show()
def inteAvgInNtx():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normalIncomeNtx1.txt','r',encoding='utf-8') as f:
        dict1 = literal_eval(f.read())
    with open('normalIncomeNtx2.txt','r',encoding='utf-8') as f:
        dict2 = literal_eval(f.read())
    with open('normalIncomeNtx3.txt','r',encoding='utf-8') as f:
        dict3 = literal_eval(f.read())
    with open('normalIncomeNtx4.txt','r',encoding='utf-8') as f:
        dict4 = literal_eval(f.read())
    with open('normalIncomeNtx5.txt','r',encoding='utf-8') as f:
        dict5 = literal_eval(f.read())
    addr2income = {}
    for addr in addrlist:
        if addr in dict1.keys():
            addr2income[addr] = addr2income.get(addr,0) + dict1[addr]
        if addr in dict2.keys():
            addr2income[addr] = addr2income.get(addr,0) + dict2[addr]
        if addr in dict3.keys():
            addr2income[addr] = addr2income.get(addr,0) + dict3[addr]
        if addr in dict4.keys():
            addr2income[addr] = addr2income.get(addr,0) + dict4[addr]
        if addr in dict5.keys():
            addr2income[addr] = addr2income.get(addr,0) + dict5[addr]
    with open('normalInNtx.txt','w',encoding='utf-8') as f:
        print(addr2income,file=f)
def NtxAddrIncome():
    with open('normalInNtx.txt','r',encoding='utf-8') as f:
        addr2income = literal_eval(f.read())
    a = sorted(addr2income.items(), key=lambda x: x[1], reverse=True)
    with open('Ntxaddr2income.txt','w',encoding='utf-8') as f:
        print(a,file=f)
def ItxAddrIncome():
    with open('normalIncomeItx1.txt','r',encoding='utf-8') as f:
        addr2income = literal_eval(f.read())
    a = sorted(addr2income.items(), key=lambda x: x[1], reverse=True)
    with open('Itxaddr2income.txt','w',encoding='utf-8') as f:
        print(a,file=f)
def normalAddrAvgIn():
    with open('normalInNtx.txt','r',encoding='utf-8') as f:
        addr2income = literal_eval(f.read())
    #打开存储有地址到交易数量的映射文件
    with open('txtypenum.txt','w') as f:
        addr2txnum = literal_eval(f.read())
#欺诈地址的前1/3生命周期的时间戳，2/3生命周期，整个生命周期
#有的欺诈地址可能livingtime为0，因为不是所有欺诈地址都发出交易
def scamAthirdLivingtime():
    with open('addr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df1 = pandas.read_csv('ntx.csv')
    df2 = pandas.read_csv('itx.csv')
    frames = [df1, df2]
    df = pandas.concat(frames)  # 拼接所有交易
    df = df.drop_duplicates()  # 去重
    df = df.sort_values('timeStamp')  # 将所有交易按时间戳排序
    addr2stamp = {}  # 地址到交易时间戳的字典
    addr2living1 = {}
    addr2living2 = {}
    addr2living3 = {}
    for addr in addrlist:
        addr2stamp[addr] = []
        addr2living1[addr] = 0
        addr2living2[addr] = 0
        addr2living3[addr] = 0
    for index, row in df.iterrows():
        if isinstance(row['to'], str) and row['to'] in addrlist:  # 统计每个地址的时间戳
            addr2stamp[row['to']].append(row['timeStamp'])
    for addr, time in addr2stamp.items():
        time.sort()
    for addr, time in addr2stamp.items():
        if len(time) > 0:  # 直接用时间戳计算
            start = time[0]
            end = time[-1]
            livingtime = end - start
            addr2living1[addr] = livingtime / 3 + start
            addr2living2[addr] = livingtime / 3 * 2 + start
            addr2living3[addr] = livingtime
    with open('scam2athirdLivingtime1.txt','w') as f:
        print(addr2living1,file=f)
    with open('scam2athirdLivingtime2.txt','w') as f:
        print(addr2living2,file=f)
    with open('scam2athirdLivingtime3.txt','w') as f:
        print(addr2living3,file=f)

def scamAthirdIntime():
    with open('addr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df1 = pandas.read_csv('ntx.csv')
    df2 = pandas.read_csv('itx.csv')
    frames = [df1, df2]
    df = pandas.concat(frames)#拼接所有交易
    df = df.drop_duplicates()  # 去重
    df = df.sort_values('timeStamp')#将所有交易按时间戳排序
    addr2stamp = {}  # 地址到交易时间戳的字典
    addr2living1 = {}
    addr2living2 = {}
    for addr in addrlist:
        addr2stamp[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:#统计每个地址的时间戳
            addr2stamp[row['to']].append(row['timeStamp'])
    for addr, time in addr2stamp.items():
        time.sort()
    for addr, time in addr2stamp.items():
        if len(time) > 0:#直接用时间戳计算
            start = time[0]
            end = time[-1]
            livingtime = end - start
            addr2living1[addr] = livingtime / 3 + start
            addr2living2[addr] = livingtime / 3 * 2 + start
    with open('scam2athirdIntime1.txt','w') as f:
        print(addr2living1,file=f)
    with open('scam2athirdIntime2.txt','w') as f:
        print(addr2living2,file=f)

def scamAthirdOuttime():#欺诈地址1/3的生命周期，2/3生命周期，全部生命周期，错误，无论是in还是out的交易都放到一起计算生命周期
    with open('addr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df1 = pandas.read_csv('ntx.csv')
    df2 = pandas.read_csv('itx.csv')
    frames = [df1,df2]
    df = pandas.concat(frames)#拼接所有交易
    df = df.drop_duplicates()  # 去重
    df = df.sort_values('timeStamp')#将所有交易按时间戳排序
    addr2day = {}  # 地址到交易时间戳的字典
    addr2living1 = {}
    addr2living2 = {}
    for addr in addrlist:
        addr2day[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:#统计每个地址的时间戳
            addr2day[row['from']].append(row['timeStamp'])
    for addr, time in addr2day.items():  # 将时间列表排序，计算时间差
        time.sort()
    for addr, time in addr2day.items():
        if len(time) > 0:#直接用时间戳计算
            start = time[0]
            end = time[-1]
            livingtime = end - start
            addr2living1[addr] = livingtime / 3 + start
            addr2living2[addr] = livingtime / 3 * 2 + start
    with open('scam2athirdOuttime1.txt', 'w') as f:
        print(addr2living1, file=f)
    with open('scam2athirdOuttime2.txt', 'w') as f:
        print(addr2living2, file=f)
#如何筛选某个时间戳之间的交易并统计其数量，小于指定时间戳的才统计，根据爬到的交易进行统计
def scamAthirdInTxnum():
    with open('addr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('scam2athirdLivingtime1.txt','r') as f:
        addr2end1 = literal_eval(f.read())
    with open('scam2athirdLivingtime2.txt','r') as f:
        addr2end2 = literal_eval(f.read())
    df1 = pandas.read_csv('ntx.csv')
    df2 = pandas.read_csv('itx.csv')
    frames = [df1,df2]
    df = pandas.concat(frames)
    df = df.drop_duplicates()  # 去重
    df = df.sort_values('timeStamp')#统计每个地址的符合时间戳条件的交易
    addr2athirdtxnum1 = {}
    addr2athirdtxnum2 = {}
    addr2athirdtxnum3 = {}
    addr2tx = {}
    for addr in addrlist:#初始化用于统计的字典
        addr2athirdtxnum1[addr] = 0
        addr2athirdtxnum2[addr] = 0
        addr2athirdtxnum3[addr] = 0
        addr2tx[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            end1 = addr2end1[row['to']]  # 每一行地址的结束时间戳，有的欺诈地址可能只是from地址而不是to地址
            end2 = addr2end2[row['to']]
            if row['timeStamp'] < end1:
                addr2tx[row['to']].append(row['hash'])#每一个地址的满足对应条件的交易hash值列表
                addr2athirdtxnum1[row['to']] += 1
            if row['timeStamp'] >= end1 and row['timeStamp'] < end2:
                addr2athirdtxnum2[row['to']] += 1
            if row['timeStamp'] >= end2:
                addr2athirdtxnum3[row['to']] += 1
    with open('scam2athirdIntxnum1.txt','w') as f:
        print(addr2athirdtxnum1,file=f)
    with open('scam2athirdIntxnum2.txt','w') as f:
        print(addr2athirdtxnum2,file=f)
    with open('scam2athirdIntxnum3.txt','w') as f:
        print(addr2athirdtxnum3,file=f)
def scamAthirdOutTxnum():
    with open('addr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('scam2athirdLivingtime1.txt','r') as f:
        addr2end1 = literal_eval(f.read())
    with open('scam2athirdLivingtime2.txt','r') as f:
        addr2end2 = literal_eval(f.read())
    df1 = pandas.read_csv('ntx.csv')
    df2 = pandas.read_csv('itx.csv')
    frames = [df1,df2]
    df = pandas.concat(frames)
    df = df.drop_duplicates()  # 去重
    df = df.sort_values('timeStamp')#统计每个地址的符合时间戳条件的交易
    addr2athirdtxnum1 = {}
    addr2athirdtxnum2 = {}
    addr2athirdtxnum3 = {}
    addr2tx = {}
    for addr in addrlist:#初始化用于统计的字典
        addr2athirdtxnum1[addr] = 0
        addr2athirdtxnum2[addr] = 0
        addr2athirdtxnum3[addr] = 0
        addr2tx[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:
            end1 = addr2end1[row['from']]# 每一行地址的结束时间戳，有的欺诈地址可能只是from地址而不是to地址
            end2 = addr2end2[row['from']]
            if row['timeStamp'] < end1:
                addr2tx[row['from']].append(row['hash'])#每一个地址的满足对应条件的交易hash值列表
                addr2athirdtxnum1[row['from']] += 1
            if row['timeStamp'] >= end1 and row['timeStamp'] < end2:
                addr2athirdtxnum2[row['from']] += 1
            if row['timeStamp'] >= end2:
                addr2athirdtxnum3[row['from']] += 1
    with open('scam2athirdOuttxnum1.txt','w') as f:
        print(addr2athirdtxnum1,file=f)
    with open('scam2athirdOuttxnum2.txt','w') as f:
        print(addr2athirdtxnum2,file=f)
    with open('scam2athirdOuttxnum3.txt','w') as f:
        print(addr2athirdtxnum3,file=f)
def norAddrA3LivingTime():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read()) # 读取去重交易后的正常地址交易
    df1 = pandas.read_csv('normalAddrntx2.csv')
    df2 = pandas.read_csv('normalAddritx2.csv')
    frames = [df1, df2]
    df = pandas.concat(frames)
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2stamp = {}  # 地址到交易时间戳的字典
    addr2living1 = {}
    addr2living2 = {}
    addr2living3 = {}
    for addr in addrlist:
        addr2stamp[addr] = []
        addr2living1[addr] = 0
        addr2living2[addr] = 0
        addr2living3[addr] = 0
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:#统计每个地址的时间戳
            addr2stamp[row['to']].append(row['timeStamp'])
    for addr, time in addr2stamp.items():
        time.sort()
    for addr, time in addr2stamp.items():
        if len(time) > 0:#直接用时间戳计算
            start = time[0]
            end = time[-1]
            livingtime = end - start
            addr2living1[addr] = livingtime / 3 + start
            addr2living2[addr] = livingtime / 3 * 2 + start
            addr2living3[addr] = livingtime
    with open('normal2athirdtime1.txt','w') as f:
        print(addr2living1,file=f)
    with open('normal2athirdtime2.txt','w') as f:
        print(addr2living2,file=f)
    with open('normal2athirdtime3.txt','w') as f:
        print(addr2living3,file=f)
#后面的生命周期计算错误
def norAddrA3InNtxTime():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv')  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2stamp = {}  # 地址到交易时间戳的字典
    addr2living1 = {}
    addr2living2 = {}
    for addr in addrlist:
        addr2stamp[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:#统计每个地址的时间戳
            addr2stamp[row['to']].append(row['timeStamp'])
    for addr, time in addr2stamp.items():
        time.sort()
    for addr, time in addr2stamp.items():
        if len(time) > 0:#直接用时间戳计算
            start = time[0]
            end = time[-1]
            livingtime = end - start
            addr2living1[addr] = livingtime / 3 + start
            addr2living2[addr] = livingtime / 3 * 2 + start
    with open('normal2athirdtimeInNtx1.txt','w') as f:
        print(addr2living1,file=f)
    with open('normal2athirdtimeInNtx2.txt','w') as f:
        print(addr2living2,file=f)
def norAddrA3InNtxTime1():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv').iloc[0:3000000]  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2stamp = {}  # 地址到交易时间戳的字典
    addr2living1 = {}
    addr2living2 = {}
    for addr in addrlist:
        addr2stamp[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:#统计每个地址的时间戳
            addr2stamp[row['to']].append(row['timeStamp'])
    for addr, time in addr2stamp.items():
        time.sort()
    for addr, time in addr2stamp.items():
        if len(time) > 0:#直接用时间戳计算
            start = time[0]
            end = time[-1]
            livingtime = end - start
            addr2living1[addr] = livingtime / 3 + start
            addr2living2[addr] = livingtime / 3 * 2 + start
    with open('normal2athirdtimeInNtx11.txt','w') as f:
        print(addr2living1,file=f)
    with open('normal2athirdtimeInNtx12.txt','w') as f:
        print(addr2living2,file=f)
def norAddrA3InNtxTime2():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv').iloc[3000000:6000000]  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2stamp = {}  # 地址到交易时间戳的字典
    addr2living1 = {}
    addr2living2 = {}
    for addr in addrlist:
        addr2stamp[addr] = []
    for addr, time in addr2stamp.items():
        time.sort()
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:  # 统计每个地址的时间戳
            addr2stamp[row['to']].append(row['timeStamp'])
    for addr, time in addr2stamp.items():
        if len(time) > 0:  # 直接用时间戳计算
            start = time[0]
            end = time[-1]
            livingtime = end - start
            addr2living1[addr] = livingtime / 3 + start
            addr2living2[addr] = livingtime / 3 * 2 + start
    with open('normal2athirdtimeInNtx21.txt', 'w') as f:
        print(addr2living1, file=f)
    with open('normal2athirdtimeInNtx22.txt', 'w') as f:
        print(addr2living2, file=f)
def norAddrA3InNtxTime3():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv').iloc[6000000:9000000]  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2stamp = {}  # 地址到交易时间戳的字典
    addr2living1 = {}
    addr2living2 = {}
    for addr in addrlist:
        addr2stamp[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:  # 统计每个地址的时间戳
            addr2stamp[row['to']].append(row['timeStamp'])
    for addr, time in addr2stamp.items():
        time.sort()
    for addr, time in addr2stamp.items():
        if len(time) > 0:  # 直接用时间戳计算
            start = time[0]
            end = time[-1]
            livingtime = end - start
            addr2living1[addr] = livingtime / 3 + start
            addr2living2[addr] = livingtime / 3 * 2 + start
    with open('normal2athirdtimeInNtx31.txt', 'w') as f:
        print(addr2living1, file=f)
    with open('normal2athirdtimeInNtx32.txt', 'w') as f:
        print(addr2living2, file=f)
def norAddrA3InNtxTime4():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv').iloc[9000000:12000000]  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2stamp = {}  # 地址到交易时间戳的字典
    addr2living1 = {}
    addr2living2 = {}
    for addr in addrlist:
        addr2stamp[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:  # 统计每个地址的时间戳
            addr2stamp[row['to']].append(row['timeStamp'])
    for addr, time in addr2stamp.items():
        time.sort()
    for addr, time in addr2stamp.items():
        if len(time) > 0:  # 直接用时间戳计算
            start = time[0]
            end = time[-1]
            livingtime = end - start
            addr2living1[addr] = livingtime / 3 + start
            addr2living2[addr] = livingtime / 3 * 2 + start
    with open('normal2athirdtimeInNtx41.txt', 'w') as f:
        print(addr2living1, file=f)
    with open('normal2athirdtimeInNtx42.txt', 'w') as f:
        print(addr2living2, file=f)
def norAddrA3InNtxTime5():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv').iloc[12000000:]  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2stamp = {}  # 地址到交易时间戳的字典
    addr2living1 = {}
    addr2living2 = {}
    for addr in addrlist:
        addr2stamp[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:  # 统计每个地址的时间戳
            addr2stamp[row['to']].append(row['timeStamp'])
    for addr, time in addr2stamp.items():
        time.sort()
    for addr, time in addr2stamp.items():
        if len(time) > 0:  # 直接用时间戳计算
            start = time[0]
            end = time[-1]
            livingtime = end - start
            addr2living1[addr] = livingtime / 3 + start
            addr2living2[addr] = livingtime / 3 * 2 + start
    with open('normal2athirdtimeInNtx51.txt', 'w') as f:
        print(addr2living1, file=f)
    with open('normal2athirdtimeInNtx52.txt', 'w') as f:
        print(addr2living2, file=f)
def norAddrA3OutNtxTime():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv')  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2stamp = {}  # 地址到交易时间戳的字典
    for addr in addrlist:
        addr2stamp[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:#统计每个地址的时间戳
            addr2stamp[row['from']].append(row['timeStamp'])
    for addr, time in addr2stamp.items():
        time.sort()
    addr2living1 = {}
    addr2living2 = {}
    for addr, time in addr2stamp.items():
        if len(time) > 0:#直接用时间戳计算
            start = time[0]
            end = time[-1]
            livingtime = end - start
            addr2living1[addr] = livingtime / 3 + start
            addr2living2[addr] = livingtime / 3 * 2 + start
    with open('normal2athirdtimeOutNtx1.txt','w') as f:
        print(addr2living1,file=f)
    with open('normal2athirdtimeOutNtx2.txt','w') as f:
        print(addr2living2,file=f)
def norAddrA3OutNtxTime1():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv').iloc[0:3000000]  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2stamp = {}  # 地址到交易时间戳的字典
    for addr in addrlist:
        addr2stamp[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:#统计每个地址的时间戳
            addr2stamp[row['from']].append(row['timeStamp'])
    for addr, time in addr2stamp.items():
        time.sort()
    addr2living1 = {}
    addr2living2 = {}
    for addr, time in addr2stamp.items():
        if len(time) > 0:#直接用时间戳计算
            start = time[0]
            end = time[-1]
            livingtime = end - start
            addr2living1[addr] = livingtime / 3 + start
            addr2living2[addr] = livingtime / 3 * 2 + start
    with open('normal2athirdtimeOutNtx11.txt','w') as f:
        print(addr2living1,file=f)
    with open('normal2athirdtimeOutNtx12.txt','w') as f:
        print(addr2living2,file=f)
def norAddrA3OutNtxTime2():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv').iloc[3000000:6000000]  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2stamp = {}  # 地址到交易时间戳的字典
    # addr2living = {}
    for addr in addrlist:
        addr2stamp[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:#统计每个地址的时间戳
            addr2stamp[row['from']].append(row['timeStamp'])
    for addr, time in addr2stamp.items():
        time.sort()
    addr2living1 = {}
    addr2living2 = {}
    for addr, time in addr2stamp.items():
        if len(time) > 0:  # 直接用时间戳计算
            start = time[0]
            end = time[-1]
            livingtime = end - start
            addr2living1[addr] = livingtime / 3 + start
            addr2living2[addr] = livingtime / 3 * 2 + start
    with open('normal2athirdtimeOutNtx21.txt', 'w') as f:
        print(addr2living1, file=f)
    with open('normal2athirdtimeOutNtx22.txt', 'w') as f:
        print(addr2living2, file=f)
def norAddrA3OutNtxTime3():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv').iloc[6000000:9000000]  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2stamp = {}  # 地址到交易时间戳的字典
    for addr in addrlist:
        addr2stamp[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:#统计每个地址的时间戳
            addr2stamp[row['from']].append(row['timeStamp'])
    for addr, time in addr2stamp.items():
        time.sort()
    addr2living1 = {}
    addr2living2 = {}
    for addr, time in addr2stamp.items():
        if len(time) > 0:  # 直接用时间戳计算
            start = time[0]
            end = time[-1]
            livingtime = end - start
            addr2living1[addr] = livingtime / 3 + start
            addr2living2[addr] = livingtime / 3 * 2 + start
    with open('normal2athirdtimeOutNtx31.txt', 'w') as f:
        print(addr2living1, file=f)
    with open('normal2athirdtimeOutNtx32.txt', 'w') as f:
        print(addr2living2, file=f)
def norAddrA3OutNtxTime4():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv').iloc[9000000:12000000]  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2stamp = {}  # 地址到交易时间戳的字典
    for addr in addrlist:
        addr2stamp[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:#统计每个地址的时间戳
            addr2stamp[row['from']].append(row['timeStamp'])
    for addr, time in addr2stamp.items():
        time.sort()
    addr2living1 = {}
    addr2living2 = {}
    for addr, time in addr2stamp.items():
        if len(time) > 0:  # 直接用时间戳计算
            start = time[0]
            end = time[-1]
            livingtime = end - start
            addr2living1[addr] = livingtime / 3 + start
            addr2living2[addr] = livingtime / 3 * 2 + start
    with open('normal2athirdtimeOutNtx41.txt', 'w') as f:
        print(addr2living1, file=f)
    with open('normal2athirdtimeOutNtx42.txt', 'w') as f:
        print(addr2living2, file=f)
def norAddrA3OutNtxTime5():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv').iloc[12000000:]  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2stamp = {}  # 地址到交易时间戳的字典
    for addr in addrlist:
        addr2stamp[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:#统计每个地址的时间戳
            addr2stamp[row['from']].append(row['timeStamp'])
    for addr, time in addr2stamp.items():
        time.sort()
    addr2living1 = {}
    addr2living2 = {}
    for addr, time in addr2stamp.items():
        if len(time) > 0:  # 直接用时间戳计算
            start = time[0]
            end = time[-1]
            livingtime = end - start
            addr2living1[addr] = livingtime / 3 + start
            addr2living2[addr] = livingtime / 3 * 2 + start
    with open('normal2athirdtimeOutNtx51.txt', 'w') as f:
        print(addr2living1, file=f)
    with open('normal2athirdtimeOutNtx52.txt', 'w') as f:
        print(addr2living2, file=f)
def norAddrA3Intxnum():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normal2athirdtime1.txt', 'r') as f:
        addr2end1 = literal_eval(f.read())
    with open('normal2athirdtime2.txt', 'r') as f:
        addr2end2 = literal_eval(f.read())
    df1 = pandas.read_csv('normalAddrntx2.csv')
    df2 = pandas.read_csv('normalAddritx2.csv')
    frames = [df1, df2]
    df = pandas.concat(frames)
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2athirdtxnum1 = {}
    addr2athirdtxnum2 = {}
    addr2athirdtxnum3 = {}
    addr2tx = {}
    for addr in addrlist:
        addr2athirdtxnum1[addr] = 0
        addr2athirdtxnum2[addr] = 0
        addr2athirdtxnum3[addr] = 0
        addr2tx[addr] = []
    for index, row in df.iterrows():#每一个地址的满足对应条件的交易hash值列表
        if isinstance(row['to'],str) and row['to'] in addrlist:# 每一行地址的结束时间戳，有的欺诈地址可能只是from地址而不是to地址
            end1 = addr2end1[row['to']]
            end2 = addr2end2[row['to']]
            if row['timeStamp'] < end1:
                addr2tx[row['to']].append(row['hash'])
                addr2athirdtxnum1[row['to']] += 1
            if row['timeStamp'] <= end2 and row['timeStamp'] > end1:
                addr2athirdtxnum2[row['to']] += 1
            if row['timeStamp'] > end2:
                addr2athirdtxnum3[row['to']] += 1
    with open('normal2athirdIntxnum1.txt','w') as f:#正常地址前1/3生命周期的in tx num
        print(addr2athirdtxnum1,file=f)
    with open('normal2athirdIntxnum2.txt','w') as f:
        print(addr2athirdtxnum2,file=f)
    with open('normal2athirdIntxnum3.txt','w') as f:
        print(addr2athirdtxnum3,file=f)
def norAddrA3InNtxTx():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normal2athirdtimeInNtx1.txt', 'r') as f:
        addr2end1 = literal_eval(f.read())
    with open('normal2athirdtimeInNtx2.txt', 'r') as f:
        addr2end2 = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv')  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2athirdtxnum1 = {}
    addr2athirdtxnum2 = {}
    addr2athirdtxnum3 = {}
    addr2tx = {}
    for addr in addrlist:
        addr2athirdtxnum1[addr] = 0
        addr2athirdtxnum2[addr] = 0
        addr2athirdtxnum3[addr] = 0
        addr2tx[addr] = []
    for index, row in df.iterrows():#每一个地址的满足对应条件的交易hash值列表
        if isinstance(row['to'],str) and row['to'] in addrlist:# 每一行地址的结束时间戳，有的欺诈地址可能只是from地址而不是to地址
            end1 = addr2end1[row['to']]
            end2 = addr2end2[row['to']]
            if row['timeStamp'] < end1:
                addr2tx[row['to']].append(row['hash'])
                addr2athirdtxnum1[row['to']] += 1
            if row['timeStamp'] <= end2 and row['timeStamp'] > end1:
                addr2athirdtxnum2[row['to']] += 1
            if row['timeStamp'] > end2:
                addr2athirdtxnum3[row['to']] += 1
    with open('normal2athirdInNtxnum1.txt','w') as f:
        print(addr2athirdtxnum1,file=f)
    with open('normal2athirdInNtxnum2.txt','w') as f:
        print(addr2athirdtxnum2,file=f)
    with open('normal2athirdInNtxnum3.txt','w') as f:
        print(addr2athirdtxnum3,file=f)
def norAddrA3InNtxTx1():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normal2athirdtimeInNtx11.txt', 'r') as f:
        addr2end1 = literal_eval(f.read())
    with open('normal2athirdtimeInNtx12.txt', 'r') as f:
        addr2end2 = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv')  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2athirdtxnum1 = {}
    addr2athirdtxnum2 = {}
    addr2athirdtxnum3 = {}
    addr2tx = {}
    for addr in addrlist:
        addr2athirdtxnum1[addr] = 0
        addr2athirdtxnum2[addr] = 0
        addr2athirdtxnum3[addr] = 0
        addr2tx[addr] = []
    for index, row in df.iterrows():#每一个地址的满足对应条件的交易hash值列表
        if isinstance(row['to'],str) and row['to'] in addrlist:# 每一行地址的结束时间戳，有的欺诈地址可能只是from地址而不是to地址
            end1 = addr2end1[row['to']]
            end2 = addr2end2[row['to']]
            if row['timeStamp'] < end1:
                addr2tx[row['to']].append(row['hash'])
                addr2athirdtxnum1[row['to']] += 1
            if row['timeStamp'] <= end2 and row['timeStamp'] > end1:
                addr2athirdtxnum2[row['to']] += 1
            if row['timeStamp'] > end2:
                addr2athirdtxnum3[row['to']] += 1
    with open('normal2athirdInNtxnum11.txt','w') as f:
        print(addr2athirdtxnum1,file=f)
    with open('normal2athirdInNtxnum12.txt','w') as f:
        print(addr2athirdtxnum2,file=f)
    with open('normal2athirdInNtxnum13.txt','w') as f:
        print(addr2athirdtxnum3,file=f)
def norAddrA3InNtxTx2():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normal2athirdtimeInNtx21.txt', 'r') as f:
        addr2end1 = literal_eval(f.read())
    with open('normal2athirdtimeInNtx22.txt', 'r') as f:
        addr2end2 = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv')
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2athirdtxnum1 = {}
    addr2athirdtxnum2 = {}
    addr2athirdtxnum3 = {}
    addr2tx = {}
    for addr in addrlist:
        addr2athirdtxnum1[addr] = 0
        addr2athirdtxnum2[addr] = 0
        addr2athirdtxnum3[addr] = 0
        addr2tx[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            end1 = addr2end1[row['to']]
            end2 = addr2end2[row['to']]
            if row['timeStamp'] < end1:
                addr2tx[row['to']].append(row['hash'])
                addr2athirdtxnum1[row['to']] += 1
            if row['timeStamp'] <= end2 and row['timeStamp'] > end1:
                addr2athirdtxnum2[row['to']] += 1
            if row['timeStamp'] > end2:
                addr2athirdtxnum3[row['to']] += 1
    with open('normal2athirdInNtxnum21.txt', 'w') as f:
        print(addr2athirdtxnum1, file=f)
    with open('normal2athirdInNtxnum22.txt', 'w') as f:
        print(addr2athirdtxnum2, file=f)
    with open('normal2athirdInNtxnum23.txt', 'w') as f:
        print(addr2athirdtxnum3, file=f)
def norAddrA3InNtxTx3():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normal2athirdtimeInNtx31.txt', 'r') as f:
        addr2end1 = literal_eval(f.read())
    with open('normal2athirdtimeInNtx32.txt', 'r') as f:
        addr2end2 = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv')
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2athirdtxnum1 = {}
    addr2athirdtxnum2 = {}
    addr2athirdtxnum3 = {}
    addr2tx = {}
    for addr in addrlist:
        addr2athirdtxnum1[addr] = 0
        addr2athirdtxnum2[addr] = 0
        addr2athirdtxnum3[addr] = 0
        addr2tx[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'], str) and row['to'] in addrlist:
            end1 = addr2end1[row['to']]
            end2 = addr2end2[row['to']]
            if row['timeStamp'] < end1:
                addr2tx[row['to']].append(row['hash'])
                addr2athirdtxnum1[row['to']] += 1
            if row['timeStamp'] <= end2 and row['timeStamp'] > end1:
                addr2athirdtxnum2[row['to']] += 1
            if row['timeStamp'] > end2:
                addr2athirdtxnum3[row['to']] += 1
    with open('normal2athirdInNtxnum31.txt', 'w') as f:
        print(addr2athirdtxnum1, file=f)
    with open('normal2athirdInNtxnum32.txt', 'w') as f:
        print(addr2athirdtxnum2, file=f)
    with open('normal2athirdInNtxnum33.txt', 'w') as f:
        print(addr2athirdtxnum3, file=f)
def norAddrA3InNtxTx4():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normal2athirdtimeInNtx41.txt', 'r') as f:
        addr2end1 = literal_eval(f.read())
    with open('normal2athirdtimeInNtx42.txt', 'r') as f:
        addr2end2 = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv')
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2athirdtxnum1 = {}
    addr2athirdtxnum2 = {}
    addr2athirdtxnum3 = {}
    addr2tx = {}
    for addr in addrlist:
        addr2athirdtxnum1[addr] = 0
        addr2athirdtxnum2[addr] = 0
        addr2athirdtxnum3[addr] = 0
        addr2tx[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'], str) and row['to'] in addrlist:
            end1 = addr2end1[row['to']]
            end2 = addr2end2[row['to']]
            if row['timeStamp'] < end1:
                addr2tx[row['to']].append(row['hash'])
                addr2athirdtxnum1[row['to']] += 1
            if row['timeStamp'] <= end2 and row['timeStamp'] > end1:
                addr2athirdtxnum2[row['to']] += 1
            if row['timeStamp'] > end2:
                addr2athirdtxnum3[row['to']] += 1
    with open('normal2athirdInNtxnum41.txt', 'w') as f:
        print(addr2athirdtxnum1, file=f)
    with open('normal2athirdInNtxnum42.txt', 'w') as f:
        print(addr2athirdtxnum2, file=f)
    with open('normal2athirdInNtxnum43.txt', 'w') as f:
        print(addr2athirdtxnum3, file=f)
def norAddrA3InNtxTx5():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normal2athirdtimeInNtx51.txt', 'r') as f:
        addr2end1 = literal_eval(f.read())
    with open('normal2athirdtimeInNtx52.txt', 'r') as f:
        addr2end2 = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv')
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2athirdtxnum1 = {}
    addr2athirdtxnum2 = {}
    addr2athirdtxnum3 = {}
    addr2tx = {}
    for addr in addrlist:
        addr2athirdtxnum1[addr] = 0
        addr2athirdtxnum2[addr] = 0
        addr2athirdtxnum3[addr] = 0
        addr2tx[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'], str) and row['to'] in addrlist:
            end1 = addr2end1[row['to']]
            end2 = addr2end2[row['to']]
            if row['timeStamp'] < end1:
                addr2tx[row['to']].append(row['hash'])
                addr2athirdtxnum1[row['to']] += 1
            if row['timeStamp'] <= end2 and row['timeStamp'] > end1:
                addr2athirdtxnum2[row['to']] += 1
            if row['timeStamp'] > end2:
                addr2athirdtxnum3[row['to']] += 1
    with open('normal2athirdInNtxnum51.txt', 'w') as f:
        print(addr2athirdtxnum1, file=f)
    with open('normal2athirdInNtxnum52.txt', 'w') as f:
        print(addr2athirdtxnum2, file=f)
    with open('normal2athirdInNtxnum53.txt', 'w') as f:
        print(addr2athirdtxnum3, file=f)
def norAddrA3Outtxnum():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normal2athirdtime1.txt', 'r') as f:
        addr2end1 = literal_eval(f.read())
    with open('normal2athirdtime2.txt', 'r') as f:
        addr2end2 = literal_eval(f.read())
    df1 = pandas.read_csv('normalAddrntx2.csv')
    df2 = pandas.read_csv('normalAddritx2.csv')
    frames = [df1, df2]
    df = pandas.concat(frames)
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2athirdtxnum1 = {}
    addr2athirdtxnum2 = {}
    addr2athirdtxnum3 = {}
    addr2tx = {}
    for addr in addrlist:  # 初始化用于统计的字典
        addr2athirdtxnum1[addr] = 0
        addr2athirdtxnum2[addr] = 0
        addr2athirdtxnum3[addr] = 0
        addr2tx[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:
            end1 = addr2end1[row['from']]
            end2 = addr2end2[row['from']]
            if row['timeStamp'] < end1:
                addr2tx[row['from']].append(row['hash'])
                addr2athirdtxnum1[row['from']] += 1
            if row['timeStamp'] >= end1 and row['timeStamp'] < end2:
                addr2athirdtxnum2[row['from']] += 1
            if row['timeStamp'] >= end2:
                addr2athirdtxnum3[row['from']] += 1
    with open('normal2athirdOuttxnum1.txt','w') as f:
        print(addr2athirdtxnum1,file=f)
    with open('normal2athirdOuttxnum2.txt','w') as f:
        print(addr2athirdtxnum2,file=f)
    with open('normal2athirdOuttxnum3.txt','w') as f:
        print(addr2athirdtxnum3,file=f)
def norAddrA3OutNtxTx():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normal2athirdtimeOutNtx1.txt', 'r') as f:
        addr2end1 = literal_eval(f.read())
    with open('normal2athirdtimeOutNtx2.txt', 'r') as f:
        addr2end2 = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv')  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2athirdtxnum1 = {}
    addr2athirdtxnum2 = {}
    addr2athirdtxnum3 = {}
    addr2tx = {}
    for addr in addrlist:  # 初始化用于统计的字典
        addr2athirdtxnum1[addr] = 0
        addr2athirdtxnum2[addr] = 0
        addr2athirdtxnum3[addr] = 0
        addr2tx[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:
            end1 = addr2end1[row['from']]
            end2 = addr2end2[row['from']]
            if row['timeStamp'] < end1:
                addr2tx[row['from']].append(row['hash'])
                addr2athirdtxnum1[row['from']] += 1
            if row['timeStamp'] >= end1 and row['timeStamp'] < end2:
                addr2athirdtxnum2[row['from']] += 1
            if row['timeStamp'] >= end2:
                addr2athirdtxnum3[row['from']] += 1
    with open('normal2athirdOutNtxnum1.txt','w') as f:
        print(addr2athirdtxnum1,file=f)
    with open('normal2athirdOutNtxnum2.txt','w') as f:
        print(addr2athirdtxnum2,file=f)
    with open('normal2athirdOutNtxnum3.txt','w') as f:
        print(addr2athirdtxnum3,file=f)
def norAddrA3OutNtxTx1():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normal2athirdtimeOutNtx11.txt', 'r') as f:
        addr2end1 = literal_eval(f.read())
    with open('normal2athirdtimeOutNtx12.txt', 'r') as f:
        addr2end2 = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv')  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2athirdtxnum1 = {}
    addr2athirdtxnum2 = {}
    addr2athirdtxnum3 = {}
    addr2tx = {}
    for addr in addrlist:  # 初始化用于统计的字典
        addr2athirdtxnum1[addr] = 0
        addr2athirdtxnum2[addr] = 0
        addr2athirdtxnum3[addr] = 0
        addr2tx[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:
            end1 = addr2end1[row['from']]
            end2 = addr2end2[row['from']]
            if row['timeStamp'] < end1:
                addr2tx[row['from']].append(row['hash'])
                addr2athirdtxnum1[row['from']] += 1
            if row['timeStamp'] >= end1 and row['timeStamp'] < end2:
                addr2athirdtxnum2[row['from']] += 1
            if row['timeStamp'] >= end2:
                addr2athirdtxnum3[row['from']] += 1
    with open('normal2athirdOutNtxnum11.txt','w') as f:
        print(addr2athirdtxnum1,file=f)
    with open('normal2athirdOutNtxnum12.txt','w') as f:
        print(addr2athirdtxnum2,file=f)
    with open('normal2athirdOutNtxnum13.txt','w') as f:
        print(addr2athirdtxnum3,file=f)

def norAddrA3OutNtxTx2():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normal2athirdtimeOutNtx21.txt', 'r') as f:
        addr2end1 = literal_eval(f.read())
    with open('normal2athirdtimeOutNtx22.txt', 'r') as f:
        addr2end2 = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv')  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2athirdtxnum1 = {}
    addr2athirdtxnum2 = {}
    addr2athirdtxnum3 = {}
    addr2tx = {}
    for addr in addrlist:  # 初始化用于统计的字典
        addr2athirdtxnum1[addr] = 0
        addr2athirdtxnum2[addr] = 0
        addr2athirdtxnum3[addr] = 0
        addr2tx[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:
            end1 = addr2end1[row['from']]
            end2 = addr2end2[row['from']]
            if row['timeStamp'] < end1:
                addr2tx[row['from']].append(row['hash'])
                addr2athirdtxnum1[row['from']] += 1
            if row['timeStamp'] >= end1 and row['timeStamp'] < end2:
                addr2athirdtxnum2[row['from']] += 1
            if row['timeStamp'] >= end2:
                addr2athirdtxnum3[row['from']] += 1
    with open('normal2athirdOutNtxnum21.txt','w') as f:
        print(addr2athirdtxnum1,file=f)
    with open('normal2athirdOutNtxnum22.txt','w') as f:
        print(addr2athirdtxnum2,file=f)
    with open('normal2athirdOutNtxnum23.txt','w') as f:
        print(addr2athirdtxnum3,file=f)
def norAddrA3OutNtxTx3():

    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normal2athirdtimeOutNtx31.txt', 'r') as f:
        addr2end1 = literal_eval(f.read())
    with open('normal2athirdtimeOutNtx32.txt', 'r') as f:
        addr2end2 = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv')  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2athirdtxnum1 = {}
    addr2athirdtxnum2 = {}
    addr2athirdtxnum3 = {}
    addr2tx = {}
    for addr in addrlist:  # 初始化用于统计的字典
        addr2athirdtxnum1[addr] = 0
        addr2athirdtxnum2[addr] = 0
        addr2athirdtxnum3[addr] = 0
        addr2tx[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['from'], str) and row['from'] in addrlist:
            end1 = addr2end1[row['from']]
            end2 = addr2end2[row['from']]
            if row['timeStamp'] < end1:
                addr2tx[row['from']].append(row['hash'])
                addr2athirdtxnum1[row['from']] += 1
            if row['timeStamp'] >= end1 and row['timeStamp'] < end2:
                addr2athirdtxnum2[row['from']] += 1
            if row['timeStamp'] >= end2:
                addr2athirdtxnum3[row['from']] += 1
    with open('normal2athirdOutNtxnum31.txt', 'w') as f:
        print(addr2athirdtxnum1, file=f)
    with open('normal2athirdOutNtxnum32.txt', 'w') as f:
        print(addr2athirdtxnum2, file=f)
    with open('normal2athirdOutNtxnum33.txt', 'w') as f:
        print(addr2athirdtxnum3, file=f)
def norAddrA3OutNtxTx4():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normal2athirdtimeOutNtx41.txt', 'r') as f:
        addr2end1 = literal_eval(f.read())
    with open('normal2athirdtimeOutNtx42.txt', 'r') as f:
        addr2end2 = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv')  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2athirdtxnum1 = {}
    addr2athirdtxnum2 = {}
    addr2athirdtxnum3 = {}
    addr2tx = {}
    for addr in addrlist:  # 初始化用于统计的字典
        addr2athirdtxnum1[addr] = 0
        addr2athirdtxnum2[addr] = 0
        addr2athirdtxnum3[addr] = 0
        addr2tx[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:
            end1 = addr2end1[row['from']]
            end2 = addr2end2[row['from']]
            if row['timeStamp'] < end1:
                addr2tx[row['from']].append(row['hash'])
                addr2athirdtxnum1[row['from']] += 1
            if row['timeStamp'] >= end1 and row['timeStamp'] < end2:
                addr2athirdtxnum2[row['from']] += 1
            if row['timeStamp'] >= end2:
                addr2athirdtxnum3[row['from']] += 1
    with open('normal2athirdOutNtxnum41.txt','w') as f:
        print(addr2athirdtxnum1,file=f)
    with open('normal2athirdOutNtxnum42.txt','w') as f:
        print(addr2athirdtxnum2,file=f)
    with open('normal2athirdOutNtxnum43.txt','w') as f:
        print(addr2athirdtxnum3,file=f)
def norAddrA3OutNtxTx5():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normal2athirdtimeOutNtx51.txt', 'r') as f:
        addr2end1 = literal_eval(f.read())
    with open('normal2athirdtimeOutNtx52.txt', 'r') as f:
        addr2end2 = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv')  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2athirdtxnum1 = {}
    addr2athirdtxnum2 = {}
    addr2athirdtxnum3 = {}
    addr2tx = {}
    for addr in addrlist:  # 初始化用于统计的字典
        addr2athirdtxnum1[addr] = 0
        addr2athirdtxnum2[addr] = 0
        addr2athirdtxnum3[addr] = 0
        addr2tx[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:
            end1 = addr2end1[row['from']]
            end2 = addr2end2[row['from']]
            if row['timeStamp'] < end1:
                addr2tx[row['from']].append(row['hash'])
                addr2athirdtxnum1[row['from']] += 1
            if row['timeStamp'] >= end1 and row['timeStamp'] < end2:
                addr2athirdtxnum2[row['from']] += 1
            if row['timeStamp'] >= end2:
                addr2athirdtxnum3[row['from']] += 1
    with open('normal2athirdOutNtxnum51.txt','w') as f:
        print(addr2athirdtxnum1,file=f)
    with open('normal2athirdOutNtxnum52.txt','w') as f:
        print(addr2athirdtxnum2,file=f)
    with open('normal2athirdOutNtxnum53.txt','w') as f:
        print(addr2athirdtxnum3,file=f)
def norAddrA3InItxTime():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddritx2.csv')  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2stamp = {}  # 地址到交易时间戳的字典
    addr2living1 = {}
    addr2living2 = {}
    for addr in addrlist:
        addr2stamp[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:  # 统计每个地址的时间戳
            addr2stamp[row['to']].append(row['timeStamp'])
    for addr, time in addr2stamp.items():
        time.sort()
    for addr, time in addr2stamp.items():
        if len(time) > 0:  # 直接用时间戳计算
            start = time[0]
            end = time[-1]
            livingtime = end - start
            addr2living1[addr] = livingtime / 3 + start
            addr2living2[addr] = livingtime / 3 * 2 + start
    with open('normal2athirdtimeInItx1.txt', 'w') as f:
        print(addr2living1, file=f)
    with open('normal2athirdtimeInItx2.txt', 'w') as f:
        print(addr2living2, file=f)
def norAddrA3OutItxTime():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddritx2.csv')  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2stamp = {}  # 地址到交易时间戳的字典
    addr2living1 = {}
    addr2living2 = {}
    for addr in addrlist:
        addr2stamp[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:  # 统计每个地址的时间戳
            addr2stamp[row['from']].append(row['timeStamp'])
    for addr, time in addr2stamp.items():
        time.sort()
    for addr, time in addr2stamp.items():
        if len(time) > 0:  # 直接用时间戳计算
            start = time[0]
            end = time[-1]
            livingtime = end - start
            addr2living1[addr] = livingtime / 3 + start
            addr2living2[addr] = livingtime / 3 * 2  + start
    with open('normal2athirdtimeOutItx1.txt', 'w') as f:
        print(addr2living1, file=f)
    with open('normal2athirdtimeOutItx2.txt', 'w') as f:
        print(addr2living2, file=f)
def norAddrA3InItxTx():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normal2athirdtimeInItx1.txt', 'r') as f:
        addr2end1 = literal_eval(f.read())
    with open('normal2athirdtimeInItx2.txt', 'r') as f:
        addr2end2 = literal_eval(f.read())
    df = pandas.read_csv('normalAddritx2.csv')  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2athirdtxnum1 = {}
    addr2athirdtxnum2 = {}
    addr2athirdtxnum3 = {}
    addr2tx = {}
    for addr in addrlist:  # 初始化用于统计的字典
        addr2athirdtxnum1[addr] = 0
        addr2athirdtxnum2[addr] = 0
        addr2athirdtxnum3[addr] = 0
        addr2tx[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            end1 = addr2end1[row['to']]
            end2 = addr2end2[row['to']]
            if row['timeStamp'] < end1:
                addr2tx[row['to']].append(row['hash'])
                addr2athirdtxnum1[row['to']] += 1
            if row['timeStamp'] >= end1 and row['timeStamp'] < end2:
                addr2athirdtxnum2[row['to']] += 1
            if row['timeStamp'] >= end2:
                addr2athirdtxnum3[row['to']] += 1
    return
    with open('normal2athirdInItxnum1.txt', 'w') as f:
        print(addr2athirdtxnum1, file=f)
    with open('normal2athirdInItxnum2.txt', 'w') as f:
        print(addr2athirdtxnum2, file=f)
    with open('normal2athirdInItxnum3.txt', 'w') as f:
        print(addr2athirdtxnum3, file=f)
def norAddrA3OutItxTx():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normal2athirdtimeOutItx1.txt', 'r') as f:
        addr2end1 = literal_eval(f.read())
    with open('normal2athirdtimeOutItx2.txt', 'r') as f:
        addr2end2 = literal_eval(f.read())
    df = pandas.read_csv('normalAddritx2.csv')  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2athirdtxnum1 = {}
    addr2athirdtxnum2 = {}
    addr2athirdtxnum3 = {}
    for addr in addrlist:
        addr2athirdtxnum1[addr] = 0
        addr2athirdtxnum2[addr] = 0
        addr2athirdtxnum3[addr] = 0
    for index, row in df.iterrows():
        if isinstance(row['from'], str) and row['from'] in addrlist:
            end1 = addr2end1[row['from']]
            end2 = addr2end2[row['from']]
            if row['timeStamp'] < end1:
                addr2athirdtxnum1[row['from']] += 1
            if row['timeStamp'] >= end1 and row['timeStamp'] < end2:
                addr2athirdtxnum2[row['from']] += 1
            if row['timeStamp'] >= end2:
                addr2athirdtxnum3[row['from']] += 1
    with open('normal2athirdOutItxnum1.txt', 'w') as f:
        print(addr2athirdtxnum1, file=f)
    with open('normal2athirdOutItxnum2.txt', 'w') as f:
        print(addr2athirdtxnum2, file=f)
    with open('normal2athirdOutItxnum3.txt', 'w') as f:
        print(addr2athirdtxnum3, file=f)
def norAddr3IOTxnum():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normal2athirdInItxnum1.txt', 'r') as f:
        normal2athirdInItxnum1 = literal_eval(f.read())
    with open('normal2athirdInItxnum2.txt', 'r') as f:
        normal2athirdInItxnum2 = literal_eval(f.read())
    with open('normal2athirdInItxnum3.txt', 'r') as f:
        normal2athirdInItxnum3 = literal_eval(f.read())

    with open('normal2athirdOutItxnum1.txt', 'r') as f:
        normal2athirdOutItxnum1 = literal_eval(f.read())
    with open('normal2athirdOutItxnum2.txt', 'r') as f:
        normal2athirdOutItxnum2 = literal_eval(f.read())
    with open('normal2athirdOutItxnum3.txt', 'r') as f:
        normal2athirdOutItxnum3 = literal_eval(f.read())

    with open('normal2athirdInNtxnum1.txt', 'r') as f:
        normal2athirdInNtxnum1 = literal_eval(f.read())
    with open('normal2athirdInNtxnum2.txt', 'r') as f:
        normal2athirdInNtxnum2 = literal_eval(f.read())
    with open('normal2athirdInNtxnum3.txt', 'r') as f:
        normal2athirdInNtxnum3 = literal_eval(f.read())

    with open('normal2athirdOutNtxnum1.txt', 'r') as f:
        normal2athirdOutNtxnum1 = literal_eval(f.read())
    with open('normal2athirdOutNtxnum2.txt', 'r') as f:
        normal2athirdOutNtxnum2 = literal_eval(f.read())
    with open('normal2athirdOutNtxnum3.txt', 'r') as f:
        normal2athirdOutNtxnum3 = literal_eval(f.read())
    normal2athirdIntxnum = {}
    normal2athirdIntxnum1 = {}
    normal2athirdIntxnum2 = {}
    normal2athirdIntxnum3 = {}
    normal2athirdOuttxnum = {}
    normal2athirdOuttxnum1 = {}
    normal2athirdOuttxnum2 = {}
    normal2athirdOuttxnum3 = {}
    for addr in addrlist:
        normal2athirdIntxnum[addr] = 0
        normal2athirdIntxnum1[addr] = 0
        normal2athirdIntxnum2[addr] = 0
        normal2athirdIntxnum3[addr] = 0
        normal2athirdOuttxnum[addr] = 0
        normal2athirdOuttxnum1[addr] = 0
        normal2athirdOuttxnum2[addr] = 0
        normal2athirdOuttxnum3[addr] = 0
    for addr in addrlist:
        if addr in normal2athirdInItxnum1.keys():
            normal2athirdIntxnum1[addr] += normal2athirdInItxnum1[addr]
            normal2athirdIntxnum[addr] += normal2athirdInItxnum1[addr]
        if addr in normal2athirdInNtxnum1.keys():
            normal2athirdIntxnum1[addr] += normal2athirdInNtxnum1[addr]
            normal2athirdIntxnum[addr] += normal2athirdInNtxnum1[addr]
        if addr in normal2athirdInItxnum2.keys():
            normal2athirdIntxnum2[addr] += normal2athirdInItxnum2[addr]
            normal2athirdIntxnum[addr] += normal2athirdInItxnum2[addr]
        if addr in normal2athirdInNtxnum2.keys():
            normal2athirdIntxnum2[addr] += normal2athirdInNtxnum2[addr]
            normal2athirdIntxnum[addr] += normal2athirdInNtxnum2[addr]
        if addr in normal2athirdInItxnum3.keys():
            normal2athirdIntxnum3[addr] += normal2athirdInItxnum3[addr]
            normal2athirdIntxnum[addr] += normal2athirdInItxnum3[addr]
        if addr in normal2athirdInNtxnum3.keys():
            normal2athirdIntxnum3[addr] += normal2athirdInNtxnum3[addr]
            normal2athirdIntxnum[addr] += normal2athirdInNtxnum3[addr]

        if addr in normal2athirdOutItxnum1.keys():
            normal2athirdOuttxnum1[addr] += normal2athirdOutItxnum1[addr]
            normal2athirdOuttxnum[addr] += normal2athirdOutItxnum1[addr]
        if addr in normal2athirdOutNtxnum1.keys():
            normal2athirdOuttxnum1[addr] += normal2athirdOutNtxnum1[addr]
            normal2athirdOuttxnum[addr] += normal2athirdOutNtxnum1[addr]
        if addr in normal2athirdOutItxnum2.keys():
            normal2athirdOuttxnum2[addr] += normal2athirdOutItxnum2[addr]
            normal2athirdOuttxnum[addr] += normal2athirdOutItxnum2[addr]
        if addr in normal2athirdOutNtxnum2.keys():
            normal2athirdOuttxnum2[addr] += normal2athirdOutNtxnum2[addr]
            normal2athirdOuttxnum[addr] += normal2athirdOutNtxnum2[addr]
        if addr in normal2athirdOutItxnum3.keys():
            normal2athirdOuttxnum3[addr] += normal2athirdOutItxnum3[addr]
            normal2athirdOuttxnum[addr] += normal2athirdOutItxnum3[addr]
        if addr in normal2athirdOutNtxnum3.keys():
            normal2athirdOuttxnum3[addr] += normal2athirdOutNtxnum3[addr]
            normal2athirdOuttxnum[addr] += normal2athirdOutNtxnum3[addr]
    with open('normal2athirdIntxnum.txt', 'w') as f:#正常地址总的in tx num
        print(normal2athirdIntxnum, file=f)
    with open('normal2athirdIntxnum1.txt', 'w') as f:
        print(normal2athirdIntxnum1, file=f)
    with open('normal2athirdIntxnum2.txt', 'w') as f:
        print(normal2athirdIntxnum2, file=f)
    with open('normal2athirdIntxnum3.txt', 'w') as f:
        print(normal2athirdIntxnum3, file=f)

    with open('normal2athirdOuttxnum.txt', 'w') as f:
        print(normal2athirdOuttxnum, file=f)
    with open('normal2athirdOuttxnum1.txt', 'w') as f:
        print(normal2athirdOuttxnum1, file=f)
    with open('normal2athirdOuttxnum2.txt', 'w') as f:
        print(normal2athirdOuttxnum2, file=f)
    with open('normal2athirdOuttxnum3.txt', 'w') as f:
        print(normal2athirdOuttxnum3, file=f)

def norAddrA3InEtxTime1():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddretx2.csv')  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2stamp = {}  # 地址到交易时间戳的字典
    addr2living = {}
    for addr in addrlist:
        addr2stamp[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:  # 统计每个地址的时间戳
            addr2stamp[row['to']].append(row['timeStamp'])
    for addr, time in addr2stamp.items():
        if len(time) > 0:  # 直接用时间戳计算
            start = time[0]
            end = time[-1]
            livingtime = end - start
            addr2living[addr] = livingtime / 3 + start
    with open('normal2athirdtimeInEtx.txt', 'w') as f:
        print(addr2living, file=f)

def norAddrA3OutEtxTime1():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddretx2.csv')  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2stamp = {}  # 地址到交易时间戳的字典
    addr2living = {}
    for addr in addrlist:
        addr2stamp[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:  # 统计每个地址的时间戳
            addr2stamp[row['from']].append(row['timeStamp'])
    for addr, time in addr2stamp.items():
        if len(time) > 0:  # 直接用时间戳计算
            start = time[0]
            end = time[-1]
            livingtime = end - start
            addr2living[addr] = livingtime / 3 + start
    with open('normal2athirdtimeOutEtx.txt', 'w') as f:
        print(addr2living, file=f)
def norAddrA3InEtxTx1():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normal2athirdtimeInEtx.txt', 'r') as f:
        addr2end = literal_eval(f.read())
    df = pandas.read_csv('normalAddretx2.csv')  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2athirdtxnum = {}
    addr2tx = {}
    for addr in addrlist:  # 初始化用于统计的字典
        addr2athirdtxnum[addr] = 0
        addr2tx[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            end = addr2end[row['to']]# 每一行地址的结束时间戳，有的欺诈地址可能只是from地址而不是to地址
            if row['timeStamp'] < end:
                addr2tx[row['to']].append(row['hash'])#每一个地址的满足对应条件的交易hash值列表
                addr2athirdtxnum[row['to']] += 1
    with open('normal2athirdInEtxnum.txt','w') as f:
        print(addr2athirdtxnum,file=f)
def norAddrA3OutEtxTx1():
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normal2athirdtimeOutEtx.txt', 'r') as f:
        addr2end = literal_eval(f.read())
    df = pandas.read_csv('normalAddretx2.csv')  # 读取去重交易后的正常地址交易
    df = df.drop_duplicates()
    df = df.sort_values('timeStamp')
    addr2athirdtxnum = {}
    addr2tx = {}
    for addr in addrlist:  # 初始化用于统计的字典
        addr2athirdtxnum[addr] = 0
        addr2tx[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['from'],str) and row['from'] in addrlist:
            end = addr2end[row['to']]# 每一行地址的结束时间戳，有的欺诈地址可能只是from地址而不是to地址
            if row['timeStamp'] < end:
                addr2tx[row['from']].append(row['hash'])#每一个地址的满足对应条件的交易hash值列表
                addr2athirdtxnum[row['from']] += 1
    with open('normal2athirdOutEtxnum.txt','w') as f:
        print(addr2athirdtxnum,file=f)
def test():
    # with open('ntx1.csv','r') as f:
    #     print(len(f.readlines()))
    # with open('ntx2.csv','r') as f:
    #     print(len(f.readlines()))
    # with open('ntx3.csv','r') as f:
    #     print(len(f.readlines()))
    # with open('ntx4.csv','r') as f:
    #     print(len(f.readlines()))
    # with open('itx1.csv','r') as f:
    #     print(len(f.readlines()))
    # with open('itx2.csv','r') as f:
    #     print(len(f.readlines()))
    # with open('itx3.csv','r') as f:
    #     print(len(f.readlines()))
    # with open('itx4.csv','r') as f:
    #     print(len(f.readlines()))
    # with open('etx1.csv','r',encoding='utf-8') as f:
    #     print(len(f.readlines()))
    # with open('etx2.csv', 'r', encoding='utf-8') as f:
    #     print(len(f.readlines()))
    # with open('etx3.csv','r',encoding='utf-8') as f:
    #     print(len(f.readlines()))
    # with open('etx4.csv','r',encoding='utf-8') as f:
    #     print(len(f.readlines()))
    # with open('norAddrTx.txt','r') as f:
    #     list = literal_eval(f.read())
    #     print(len(list))
    # with open('norAddrTx.csv', 'r') as f:
    #     print(len(f.readlines()))
    # with open('normalAddrntx.csv', 'r') as f:
    #     print(len(f.readlines()))
    # with open('normalAddrntx2.csv', 'r') as f:
    #     print(len(f.readlines()))
    # chunker = pandas.read_csv("normalAddrntx2.csv", nrows=7500000)
    # print(type(chunker))
    # pd = pandas.read_csv('normalAddretx2.csv')
    # print(len(pd))
    with open('NormalAddr.txt','r') as f:
        list = literal_eval(f.read())
    print(len(list))
    # print(list.index('0x8909cc8d294544ca2c956550edbcb59ce4f2a9ad'))
    # for item in chunker:
    #     print(item)
    # with open('normalAddritx.csv', 'r') as f:
    #     print(len(f.readlines()))
    # url = "https://api.etherscan.io/api?module=account&action=txlistinternal&address=0x2c1ba59d6f58433fb1eaee7d20b26ed83bda51a3&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=ZF9TQA39PFPPUD7VCDK2Q9ZVD2M72N2HGZ"
    # session = requests.Session()
    # results = literal_eval(session.get(url).text)['result']
    # print(len(results))
def mixAddrFig9():
    # with open('norAddrNFig9Intxs.txt') as f:
    #     txnum2addrnum1 = literal_eval(f.read())
    # with open('norAddrIFig9Intxs.txt') as f:
    #     txnum2addrnum2 = literal_eval(f.read())
    # zipped1 = zip(txnum2addrnum1.keys(), txnum2addrnum1.values())
    # sort_zipped1 = sorted(zipped1, key=lambda x: (x[0]))
    # result1 = zip(*sort_zipped1)
    # x1, y1 = [list(x) for x in result1]
    # fig, ax = plt.subplots()
    # cum = numpy.cumsum(y1)
    # percentage1 = cum / list(cum)[-1]
    # zipped2 = zip(txnum2addrnum2.keys(), txnum2addrnum2.values())
    # sort_zipped2 = sorted(zipped2, key=lambda x: (x[0]))
    # result2 = zip(*sort_zipped2)
    # x2, y2 = [list(x) for x in result2]
    # fig, ax = plt.subplots()
    # cum = numpy.cumsum(y2)
    # percentage2 = cum / list(cum)[-1]
    with open('scamAddrNFig9Intx.txt',encoding='utf-8') as f:
        txnum2addrnum3 = literal_eval(f.read())
    zipped3 = zip(txnum2addrnum3.keys(), txnum2addrnum3.values())
    sort_zipped3 = sorted(zipped3, key=lambda x: (x[0]))
    result3 = zip(*sort_zipped3)
    x3, y3 = [list(x) for x in result3]
    # fig, ax = plt.subplots()
    cum = numpy.cumsum(y3)
    percentage3 = cum / list(cum)[-1]
    with open('scamAddrIFig9Intx.txt',encoding='utf-8') as f:
        txnum2addrnum4 = literal_eval(f.read())
    zipped4 = zip(txnum2addrnum4.keys(), txnum2addrnum4.values())
    sort_zipped4 = sorted(zipped4, key=lambda x: (x[0]))
    result4 = zip(*sort_zipped4)
    x4, y4 = [list(x) for x in result4]
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y4)
    percentage4 = cum / list(cum)[-1]
    # line1 = ax.plot(x1, percentage1)
    # line2 = ax.plot(x2, percentage2)
    line3 = ax.plot(x3, percentage3, label='In txs Of scam addresses',color='black')
    line4 = ax.plot(x4, percentage4, label='Out txs Of scam addresses',linestyle="--",color='black')
    ax.set_xlabel('# of Transactions (Txs)')
    ax.set_ylabel('CDF of Addresses')
    plt.xscale('log')
    plt.legend()
    plt.title('test')
    plt.show()
# def scamLive():#living time
#     with open('addr.txt', 'r', encoding='utf-8') as f:
#         addrlist = literal_eval(f.read())
#     df1 = pandas.read_csv('ntx1.csv')
#     df2 = pandas.read_csv('ntx2.csv')
#     df3 = pandas.read_csv('ntx3.csv')
#     df4 = pandas.read_csv('ntx4.csv')
#     df5 = pandas.read_csv('itx1.csv')
#     df6 = pandas.read_csv('itx2.csv')
#     df7 = pandas.read_csv('itx3.csv')
#     df8 = pandas.read_csv('itx4.csv')
#     df9 = pandas.read_csv('bntx1.csv')
#     df10 = pandas.read_csv('bitx1.csv')
#     frames = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]
#     df = pandas.concat(frames)
#     df = df.drop_duplicates()  # 去重
#     df = df.sort_values('timeStamp')
#     addr2day = {}  # 地址到交易时间戳的字典
#     addr2living = {}
#     for addr in addrlist:
#         addr2day[addr] = []
#     for index, row in df.iterrows():
#         if isinstance(row['to'],str) and row['to'] in addrlist:
#             addr2day[row['to']].append(row['timeStamp'])
#     for addr, time in addr2day.items():  # 将时间列表排序，计算时间差
#         time.sort()
#     for addr, time in addr2day.items():
#         if len(time) > 0:
#             start = datetime.datetime.fromtimestamp(time[0])
#             end = datetime.datetime.fromtimestamp(time[-1])
#             addr2living[addr] = (end - start).days
#     print(addr2living)
#     livingtime2num = {}
#     for addr,livingtime in addr2living.items():
#         livingtime2num[livingtime] = livingtime2num.get(livingtime,0) + 1
#     print(livingtime2num)
#     zipped = zip(livingtime2num.keys(), livingtime2num.values())
#     sort_zipped = sorted(zipped, key=lambda x: (x[0]))
#     result = zip(*sort_zipped)
#     x, y = [list(x) for x in result]
#     fig, ax = plt.subplots()
#     cum = numpy.cumsum(y)
#     percentage = cum / list(cum)[-1]
#     line = ax.plot(x, percentage)
#     x_major_locator = MultipleLocator(250)
#     ax.xaxis.set_major_locator(x_major_locator)
#     plt.show()
def scamLive():#living time
    with open('addr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df1 = pandas.read_csv('ntx1.csv')
    df2 = pandas.read_csv('ntx2.csv')
    df3 = pandas.read_csv('ntx3.csv')
    df4 = pandas.read_csv('ntx4.csv')
    df5 = pandas.read_csv('itx1.csv')
    df6 = pandas.read_csv('itx2.csv')
    df7 = pandas.read_csv('itx3.csv')
    df8 = pandas.read_csv('itx4.csv')
    df9 = pandas.read_csv('bntx1.csv')
    df10 = pandas.read_csv('bitx1.csv')
    df11 = pandas.read_csv('etx1.csv')
    df12 = pandas.read_csv('etx2.csv')
    df13 = pandas.read_csv('etx3.csv')
    df14 = pandas.read_csv('etx4.csv')
    df15 = pandas.read_csv('betx1.csv')
    frames = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15]
    df = pandas.concat(frames)
    df = df.drop_duplicates()  # 去重
    df = df.sort_values('timeStamp')
    addr2day = {}  # 地址到交易时间戳的字典
    addr2living = {}
    for addr in addrlist:
        addr2day[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            addr2day[row['to']].append(row['timeStamp'])
    for addr, time in addr2day.items():  # 将时间列表排序，计算时间差
        time.sort()
    for addr, time in addr2day.items():
        if len(time) > 0:
            start = datetime.datetime.fromtimestamp(time[0])
            end = datetime.datetime.fromtimestamp(time[-1])
            addr2living[addr] = (end - start).days
    livingtime2num = {}
    for addr,livingtime in addr2living.items():
        livingtime2num[livingtime] = livingtime2num.get(livingtime,0) + 1
    print(livingtime2num)
    with open('scamLive1.txt','w') as f:
        print(addr2living,file=f)
    with open('scamLive2.txt','w') as f:
        print(livingtime2num,file=f)
    return
    zipped = zip(livingtime2num.keys(), livingtime2num.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x, percentage)
    x_major_locator = MultipleLocator(250)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()
def normalLive1():#living time截取正常地址的前三分之一进行living time的统计
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddrntx2.csv').iloc[0:5000000]  # hash,from,to,value
    df = df.drop_duplicates()  # 去重
    df = df.sort_values('timeStamp')
    addr2day = {}  # 地址到交易时间戳的字典
    addr2living = {}
    for addr in addrlist:
        addr2day[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            addr2day[row['to']].append(row['timeStamp'])
    for addr, time in addr2day.items():  # 将时间列表排序，计算时间差
        time.sort()
    for addr, time in addr2day.items():
        if len(time) > 0:
            start = datetime.datetime.fromtimestamp(time[0])
            end = datetime.datetime.fromtimestamp(time[-1])
            addr2living[addr] = (end - start).days
    print(addr2living)
    livingtime2num = {}
    for addr,livingtime in addr2living.items():
        livingtime2num[livingtime] = livingtime2num.get(livingtime,0) + 1
    print(livingtime2num)
    with open('normalLiven1.txt','w') as f:
        print(addr2living,file=f)
    with open('normalLiven2.txt','w') as f:
        print(livingtime2num,file=f)
    zipped = zip(livingtime2num.keys(), livingtime2num.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x, percentage)
    x_major_locator = MultipleLocator(250)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()
def normalLive2():#living time
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddritx2.csv')  # hash,from,to,value
    df = df.drop_duplicates()  # 去重
    df = df.sort_values('timeStamp')
    addr2day = {}  # 地址到交易时间戳的字典
    addr2living = {}
    for addr in addrlist:
        addr2day[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            addr2day[row['to']].append(row['timeStamp'])
    for addr, time in addr2day.items():  # 将时间列表排序，计算时间差
        time.sort()
    for addr, time in addr2day.items():
        if len(time) > 0:
            start = datetime.datetime.fromtimestamp(time[0])
            end = datetime.datetime.fromtimestamp(time[-1])
            addr2living[addr] = (end - start).days
    # with open('normalLivei1.txt','w') as f:
    #     print(addr2living,file=f)
    livingtime2num = {}
    for addr,livingtime in addr2living.items():
        livingtime2num[livingtime] = livingtime2num.get(livingtime,0) + 1
    print(livingtime2num)
    # with open('normalLivei2.txt','w') as f:
    #     print(livingtime2num,file=f)
    zipped = zip(livingtime2num.keys(), livingtime2num.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x, percentage)
    x_major_locator = MultipleLocator(250)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()
def normalLive3():#living time
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    df = pandas.read_csv('normalAddretx2.csv')  # hash,from,to,value
    df = df.drop_duplicates()  # 去重
    df = df.sort_values('timeStamp')
    addr2day = {}  # 地址到交易时间戳的字典
    addr2living = {}
    for addr in addrlist:
        addr2day[addr] = []
    for index, row in df.iterrows():
        if isinstance(row['to'],str) and row['to'] in addrlist:
            addr2day[row['to']].append(row['timeStamp'])
    for addr, time in addr2day.items():  # 将时间列表排序，计算时间差
        time.sort()
    for addr, time in addr2day.items():
        if len(time) > 0:
            start = datetime.datetime.fromtimestamp(time[0])
            end = datetime.datetime.fromtimestamp(time[-1])
            addr2living[addr] = (end - start).days
    print(addr2living)
    livingtime2num = {}
    for addr,livingtime in addr2living.items():
        livingtime2num[livingtime] = livingtime2num.get(livingtime,0) + 1
    print(livingtime2num)
    with open('normalLivee1.txt','w') as f:
        print(addr2living,file=f)
    with open('normalLivee2.txt','w') as f:
        print(livingtime2num,file=f)
    zipped = zip(livingtime2num.keys(), livingtime2num.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x, percentage)
    x_major_locator = MultipleLocator(250)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()
def mixnormalLive():
    with open('normalLiven1.txt','r') as f:
        addr2living1 = literal_eval(f.read())
    with open('normalLivei1.txt','r') as f:
        addr2living2 = literal_eval(f.read())
    with open('normalLivee1.txt','r') as f:
        addr2living3 = literal_eval(f.read())
    #应该重新统计，有可能某个地址同时有三种交易，但是要取最长的living time，取出地址列表
    #创建一个地址到living time的总时间字典
    #因为normal tx取的是前三分之一的交易，所以地址列表应该是全部正常地址列表的子集
    addr2living = {}
    addrlist = list(addr2living1.keys()) + list(addr2living2.keys()) + list(addr2living3.keys())
    for addr in addrlist:
        addr2living[addr] = []
    for addr in addrlist:
        if addr in addr2living1.keys():
            addr2living[addr].append(addr2living1[addr])
        if addr in addr2living2.keys():
            addr2living[addr].append(addr2living2[addr])
        if addr in addr2living3.keys():
            addr2living[addr].append(addr2living3[addr])
    for addr,livinglist in addr2living.items():
        addr2living[addr] = sorted(addr2living[addr])

    livingtime2num = {}
    for addr,livingtime in addr2living.items():
        livingtime2num[livingtime[-1]] = livingtime2num.get(livingtime[-1],0) + 1
    with open('normalLive.txt','w') as f:
        print(livingtime2num,file=f)
    return
def mixLive():
    with open('normalLive.txt','r') as f:
        livingtime2num1 = literal_eval(f.read())
    with open('scamLive2.txt','r') as f:
        livingtime2num2 = literal_eval(f.read())
    zipped1 = zip(livingtime2num1.keys(), livingtime2num1.values())
    sort_zipped1 = sorted(zipped1, key=lambda x: (x[0]))
    result1 = zip(*sort_zipped1)
    x1, y1 = [list(x) for x in result1]
    zipped2 = zip(livingtime2num2.keys(), livingtime2num2.values())
    sort_zipped2 = sorted(zipped2, key=lambda x: (x[0]))
    result2 = zip(*sort_zipped2)
    x2, y2 = [list(x) for x in result2]

    fig, ax = plt.subplots()
    cum1 = numpy.cumsum(y1)
    percentage1 = cum1 / list(cum1)[-1]
    line1 = ax.plot(x1, percentage1, label="living time of normal address", color="black", linestyle="--")
    cum2 = numpy.cumsum(y2)
    percentage2 = cum2 / list(cum2)[-1]
    line2 = ax.plot(x2, percentage2, label="living time of scam address", color="black")

    x_major_locator = MultipleLocator(250)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.title('living time of addresses(Days)')
    plt.legend()
    ax.set_ylabel('CDF of Addresses')
    ax.set_xlim(1)
    plt.savefig('livingtime.jpg')
    plt.show()
def scamTxDedupN():
    df1 = pandas.read_csv('ntx1.csv')
    df2 = pandas.read_csv('ntx2.csv')
    df3 = pandas.read_csv('ntx3.csv')
    df4 = pandas.read_csv('ntx4.csv')
    df5 = pandas.read_csv('ntx5.csv')
    df6 = pandas.read_csv('bntx1.csv')
    frames = [df1, df2, df3, df4, df5, df6]
    df = pandas.concat(frames)
    df = df.drop_duplicates()
    df['hash'] = df['hash'].str.lower()#对哈希值进行全小写处理，然后去重，有可能有小写后相同的地址
    df = df.drop_duplicates(subset=['hash'], keep='first')
    df.to_csv('ntx.csv')
def scamTxDedupI():
    df1 = pandas.read_csv('itx1.csv')
    df2 = pandas.read_csv('itx2.csv')
    df3 = pandas.read_csv('itx3.csv')
    df4 = pandas.read_csv('itx4.csv')
    # df5 = pandas.read_csv('itx5.csv')
    df6 = pandas.read_csv('bitx1.csv')
    frames = [df1, df2, df3, df4, df6]
    df = pandas.concat(frames)
    df = df.drop_duplicates()
    df['hash'] = df['hash'].str.lower()
    df = df.drop_duplicates(subset=['hash'], keep='first')
    df.to_csv('itx.csv')
def scamTxDedupE():
    df1 = pandas.read_csv('etx1.csv')
    df2 = pandas.read_csv('etx2.csv')
    df3 = pandas.read_csv('etx3.csv')
    df4 = pandas.read_csv('etx4.csv')
    # df5 = pandas.read_csv('etx5.csv')
    df6 = pandas.read_csv('betx1.csv')
    frames = [df1, df2, df3, df4, df6]
    df = pandas.concat(frames)
    df = df.drop_duplicates()
    df['hash'] = df['hash'].str.lower()
    df = df.drop_duplicates(subset=['hash'], keep='first')
    df.to_csv('etx.csv')
def txdedupN():
    df = pandas.read_csv('normalAddrntx.csv')
    df['hash'] = df['hash'].str.lower()#根据小写哈希值去重
    df = df.drop_duplicates(subset=['hash'],keep='first')
    for index, row in df.iterrows():
        if isinstance(row['from'],str):
            row['from'] = row['from'].lower()
        if isinstance(row['to'],str):
            row['to'] = row['to'].lower()
    df.to_csv('normalAddrntx2.csv')
def txdedupI():
    df = pandas.read_csv('normalAddritx.csv')
    df['hash'] = df['hash'].str.lower()
    df = df.drop_duplicates(subset=['hash'],keep='first')
    for index, row in df.iterrows():
        if isinstance(row['from'],str):
            row['from'] = row['from'].lower()
        if isinstance(row['to'],str):
            row['to'] = row['to'].lower()
    df.to_csv('normalAddritx2.csv')
def txdedupE():
    df = pandas.read_csv('normalAddretx.csv')
    df['hash'] = df['hash'].str.lower()
    df = df.drop_duplicates(subset=['hash'],keep='first')
    for index, row in df.iterrows():
        if isinstance(row['from'], str):
            row['from'] = row['from'].lower()
        if isinstance(row['to'], str):
            row['to'] = row['to'].lower()
    df.to_csv('normalAddretx2.csv')
def bloxyaddrtax():
    #怎么处理bloxy的地址标记，从备注中获取，但是备注中有可能有多种欺诈，以etherscan的为主，因为bloxy的很少
    with open('bloxyhack.txt', 'r', encoding='utf-8') as f:
        dict1 = literal_eval(f.read())
    with open('bloxyscam.txt', 'r', encoding='utf-8') as f:
        dict2 = literal_eval(f.read())
    with open('bloxymalware.txt', 'r', encoding='utf-8') as f:
        dict3 = literal_eval(f.read())
    mydict = {}
    mydict.update(dict1)
    mydict.update(dict2)
    mydict.update(dict3)
    # print(len(mydict))
    hacktype = ['Malware', 'ICO', 'Trust-Trading', 'Trust Trading', 'Phish', 'Hack']
    notType = ['Hacking', 'HackToken', 'HackDao', 'HackerGold', 'Hacken', 'HackerSpaceBarneysToken', 'HackerToken', 'Gitcoin', 'VitaluckHack']
    wrongaddr = ['0x4b2b60175b6e070e456816fda2b01a72343d66ca', '0x451ab68bbec364b8a9e4403741d311c54e3907f6', '0x9e6b2b11542f2bc52f3029077ace37e8fd838d7f', '0xdcf59ee4803931a376a0fb6244036e49ebc7dd61', '0x433d84a1c0b650eac4fc1fb570798cf655ce24c0', '0xcd3e727275bc2f511822dc9a26bd7b0bbf161784', '0x2563c25db9d2fb94f65e7632544eb057fc3dfcf0', '0x2559f0dd36179c230443d2c3846435e52eb2b2df', '0x14f37b574242d366558db61f3335289a5035c506', '0x06044b5359d8df7886366c22c61c7ecd29becac7', '0xa62bdee2f277c2e2c0f46cba96879b263796ee1c', '0x05fb86775fd5c16290f1e838f5caaa7342bd9a63', '0xf6fe061efa2a8e15936696baf5e8cba8c3f3485b', '0x5c430fa24f782cf8156ca97208c42127b17b0494', '0x1f2e2293efa2ebd9c09211569f3d7758f0463189']
    typenum = {}
    typelist = {}
    addrlist = []
    #提前删除错误的地址
    for mytype in hacktype:
        typenum[mytype] = 0
        typelist[mytype] = []
    for addr in wrongaddr:
        if str('/tx/' + addr) in mydict.keys():
            mydict.pop(str('/tx/' + addr))
        elif str('/address/' + addr) in mydict.keys():
            mydict.pop(str('/address/' + addr))
    for url,note in mydict.items():
        note = note.split(",")
        exitflag = False
        for mytype in hacktype:
            for tip in note:
                if mytype.lower() in tip.lower():
                    exitflag = True
                    if url.startswith('/address'):
                        addr = url[9:]
                    elif url.startswith('/tx'):
                        addr = url[4:]
                    if len(addr) == 42 and addr not in addrlist:
                        typenum[mytype] += 1
                        addr = addr.lower()
                        typelist[mytype].append(addr)
                        addrlist.append(addr)
                    break
            if exitflag:
                break
    addrlist = list(set(addrlist))
    sum = 0
    for type,num in typenum.items():
        sum += num
    for type,list1 in typelist.items():
        list1 = list(set(list1))
    # print(sum)
    # print(len(addrlist))
    # print(typenum)
    return addrlist,typenum,typelist#bloxy的地址列表，类型统计，类型地址映射
def etherscanAddrTax():
    addrlist, typenum, typelist = bloxyaddrtax()
    with open('bitpoint.txt', 'r') as f:
        list1 = literal_eval(f.read())
        list1 = [addr.lower() for addr in list1]
        list1 = list(set(list1))
    with open('cryptopia.txt', 'r') as f:
        list2 = literal_eval(f.read())
        list2 = [addr.lower() for addr in list2]
        list2 = list(set(list2))
    with open('etherdelta.txt', 'r') as f:
        list3 = literal_eval(f.read())
        list3 = [addr.lower() for addr in list3]
        list3 = list(set(list3))
    with open('LendfMe.txt', 'r') as f:
        list4 = literal_eval(f.read())
        list4 = [addr.lower() for addr in list4]
        list4 = list(set(list4))
    with open('phishaddr.txt', 'r') as f:
        list5 = literal_eval(f.read())
        list5 = [addr.lower() for addr in list5]
        list5 = list(set(list5))#在etherscan的phish类
        # typenum['Phish']  += len(list5)#只有一个文件是phish类的，把这个文件中的bloxy地址去掉即可，另外的文件中专属于etherscan的地址全部统计到hack类中
    with open('upbit.txt', 'r') as f:
        list6 = literal_eval(f.read())
        list6 = [addr.lower() for addr in list6]
        list6 = list(set(list6))
    with open('scam.txt', 'r') as f:
        list7 = literal_eval(f.read())
        list7 = [addr.lower() for addr in list7]
        list7 = list(set(list7))
    with open('db.txt', 'r') as f:
        list8 = literal_eval(f.read())
        list8 = [addr.lower() for addr in list8]
        list8 = list(set(list8))
        print(len(list8))
    return
    with open('bloxyAddr.txt', 'r') as f:#在bloxy中的所有地址
        list9 = literal_eval(f.read())
        list9 = [addr.lower() for addr in list9]
        list9 = list(set(list9))#在bloxy的所有地址
        mylen = len(list9)
        print(mylen)
    # return
    #已经分好类的地址：在bloxy中的所有地址；未分好类的地址：在scan中而不在bloxy中的地址
    scanlist = list(set(list1 + list2 + list3 + list4 + list5 + list6 + list7))#scan中的地址数
    print(len(scanlist))
    return
    scanlist = list(set(list1 + list2 + list3 + list4 + list5 + list6 + list7 + list8))#scan中的地址数
    print(len(scanlist))#6429
    hacklist = list(set(list1 + list2 + list3 + list4 + list6 + list7 + list8))#etherscan去重后的hack类地址列表
    phishlist = list5
    bloxylist = list9
    hackandphish = [addr for addr in phishlist if addr in hacklist]
    print(len(hackandphish))#2284
    onlyscan = [addr for addr in scanlist if addr not in bloxylist]
    print(len(onlyscan))#2434
    # mylist1 = [i for i in list5 if i not in list9 and i not in hacklist]  # 在etherscan不在bloxy而且不在hack类的Phish类的地址列表
    # print(len(list9))
    # print(len(addrlist))
    phishnothack = []
    for addr in onlyscan:
        if addr not in hacklist:#phish类要去除bloxy已经分好类的，并且不能是hack类
            phishnothack.append(addr)
    phishnothack = list(set(phishnothack))#去重
    typelist['Phish'] = typelist['Phish'] + phishnothack
    typelist['Phish'] = list(set(typelist['Phish']))#地址去重
    typenum['Phish'] += len(phishnothack)#分类计数增加
    # mylist2 = [i for i in hacklist if i not in list9 and i not in list5]#在etherscan不在bloxy而且不在phish类的Hack类的地址列表
    mylist2 = []
    for addr in onlyscan:
        if addr in hacklist:#hack类要去除bloxy已经分好类的，并且不能是phish类
            mylist2.append(addr)
    mylist2 = list(set(mylist2))#去重
    typelist['Hack'] = typelist['Hack'] + mylist2
    typelist['Hack'] = list(set(typelist['Hack']))#地址去重
    typenum['Hack'] += len(mylist2)#分类计数增加
    mylist3 = []
    sum = 0#分好类的地址数
    for addr in typelist.values():#mylist3是已经分好类的地址列表
        sum += len(addr)
        mylist3 += addr
    sum2 = 0#分好类的地址数
    for num in typenum.values():#6449
        sum2 += num
    mylist3 = list(set(mylist3))#去重
    print("sum："+str(sum))#6434
    print("sum2："+str(sum2))#6449
    print(len(mylist3))#6449
    alladdr = list(set(list1 + list2 + list3 + list4 + list5 + list6 + list7 + list8 + list9))
    print(len(alladdr))#6434
    mylist5 = [addr for addr in mylist3 if addr not in alladdr]
    # print(mylist5)
    print(typenum)
    with open('addr.txt', 'r') as f:
        list10 = literal_eval(f.read())
        print(len(list10))#6434
    mylist4 = [i for i in alladdr if i not in list10]#漏掉的欺诈地址
    # print(mylist4)
    # print(typelist)
    with open('typelist.txt','w') as f:
        print(typelist,file=f)
    return typelist
# {'Malware': 2, 'ICO': 13, 'Trust-Trading': 1290, 'Trust Trading': 1, 'Phish': 3185, 'Hack': 1943}
# 不是欺诈地址的地址：
#0x1f2e2293efa2ebd9c09211569f3d7758f0463189
#0x14f37b574242d366558db61f3335289a5035c506
#['0x4b2b60175b6e070e456816fda2b01a72343d66ca', '0x451ab68bbec364b8a9e4403741d311c54e3907f6',
# '0x9e6b2b11542f2bc52f3029077ace37e8fd838d7f', '0xdcf59ee4803931a376a0fb6244036e49ebc7dd61',
# '0x433d84a1c0b650eac4fc1fb570798cf655ce24c0', '0xcd3e727275bc2f511822dc9a26bd7b0bbf161784',
# '0x2563c25db9d2fb94f65e7632544eb057fc3dfcf0', '0x2559f0dd36179c230443d2c3846435e52eb2b2df',
# '0x14f37b574242d366558db61f3335289a5035c506', '0x06044b5359d8df7886366c22c61c7ecd29becac7',
# '0xa62bdee2f277c2e2c0f46cba96879b263796ee1c', '0x05fb86775fd5c16290f1e838f5caaa7342bd9a63',
# '0xf6fe061efa2a8e15936696baf5e8cba8c3f3485b', '0x5c430fa24f782cf8156ca97208c42127b17b0494', '0x1f2e2293efa2ebd9c09211569f3d7758f0463189']
def txtax():#交易分类
    with open('typelist.txt','r') as f:
        typelist = literal_eval(f.read())
    with open('addr.txt','r') as f:
        scamaddr = literal_eval(f.read())
    print(len(scamaddr))
    return
    txtypenum = {}
    addrtxnum = {}#记录每个种类的各个地址的各种交易数量
    for addr in scamaddr:
        addrtxnum[addr] = [0,0,0]
    hacktype = ['Malware', 'ICO', 'Trust-Trading', 'Trust Trading', 'Phish', 'Hack']
    session = requests.Session()
    for mytype in hacktype:
        txtypenum[mytype] = [0,0,0]
    num = 0
    for typename,addrlist in typelist.items():
        for addr in addrlist:
            nurl = "https://api.etherscan.io/api?module=account&action=txlist&address="+addr+"&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            iurl = "https://api.etherscan.io/api?module=account&action=txlistinternal&address=" + addr + "&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            eurl = "https://api.etherscan.io/api?module=account&action=tokentx&address=" + addr + "&page=1&offset=10000&startblock=0&endblock=99999999&sort=asc&apikey=" + apikey
            result1 = literal_eval(session.get(nurl).text)['result']
            result2 = literal_eval(session.get(iurl).text)['result']
            result3 = literal_eval(session.get(eurl).text)['result']
            len1 = len(result1)
            len2 = len(result2)
            len3 = len(result3)
            txtypenum[typename][0] += len1
            txtypenum[typename][1] += len2
            txtypenum[typename][2] += len3
            addrtxnum[addr][0] += len1
            addrtxnum[addr][0] += len2
            addrtxnum[addr][0] += len3
            num += 1
            print(num)
    with open('txtypenum.txt','w') as f:
        print(txtypenum,file=f)
def filterAddr1():
    with open('nor2numNOuttx1.txt','r',encoding='utf-8') as f:
        addr2tx2 = literal_eval(f.read())
    with open('nor2numNOuttx2.txt','r',encoding='utf-8') as f:
        addr2tx3 = literal_eval(f.read())
    with open('nor2numNOuttx3.txt','r',encoding='utf-8') as f:
        addr2tx4 = literal_eval(f.read())
    addr2tx = addr2tx2 + addr2tx3 + addr2tx4
    addrlist = []
    for addr,tx in addr2tx.items():
        if tx > 20000:
            addrlist.append(addr)
    for addr in addrlist:
        addr2tx.pop(addr)
    with open('filterAddr.txt','w',encoding='utf-8') as f:
        print(list(addr2tx.keys()),file=f)
def filterAddr2():
    with open('nor2numNIntx.txt','r',encoding='utf-8') as f:
        addr2tx = literal_eval(f.read())
    addrlist = []
    for addr,tx in addr2tx.items():
        if tx > 20000:
            addrlist.append(addr)
    for addr in addrlist:
        addr2tx.pop(addr)
    with open('filterAddr.txt','w',encoding='utf-8') as f:
        print(list(addr2tx.keys()),file=f)
def scamAvgIncomeOutcome():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('scamIncome.txt','r',encoding='utf-8') as f:
        addr2income = literal_eval(f.read())
    with open('scamOutcome.txt','r',encoding='utf-8') as f:
        addr2outcome = literal_eval(f.read())
    with open('scamInTx.txt','r') as f:
        addr2Intxnum = literal_eval(f.read())
    with open('scamOutTx.txt','r',encoding='utf-8') as f:
        addr2OutTxnum = literal_eval(f.read())
    addr2avgincome = {}
    addr2avgoutcome = {}
    for addr in addrlist:
        addr2avgincome[addr] = 0
        addr2avgoutcome[addr] = 0
    for addr in addrlist:
        if addr in addr2Intxnum.keys() and addr in addr2income.keys():
            if addr2Intxnum[addr] != 0:
                addr2avgincome[addr] = addr2income[addr] / addr2Intxnum[addr]
    for addr in addrlist:
        if addr in addr2OutTxnum.keys() and addr in addr2outcome.keys():
            if addr2OutTxnum[addr] != 0:
                addr2avgoutcome[addr] = addr2outcome[addr] / addr2OutTxnum[addr]
    print(addr2avgincome)
    print(addr2avgoutcome)
    with open('scamAvgIncome.txt','w') as f:
        print(addr2avgincome,file=f)
    with open('scamAvgOutcome.txt','w') as f:
        print(addr2avgoutcome,file=f)

def norItxAvgIOcome():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normalIncomeItx1.txt','r',encoding='utf-8') as f:
        addr2income = literal_eval(f.read())
    with open('normalOutcomeItx1.txt','r',encoding='utf-8') as f:
        addr2outcome = literal_eval(f.read())
    with open('nor2numIIntx1.txt','r') as f:
        addr2Intxnum = literal_eval(f.read())
    with open('nor2numIIntx1.txt','r') as f:
        addr2Intxnum = literal_eval(f.read())
    with open('nor2numIOuttx.txt','r',encoding='utf-8') as f:
        addr2OutTxnum = literal_eval(f.read())
    addr2avgincome = {}
    addr2avgoutcome = {}
    for addr in addrlist:
        if addr in addr2Intxnum.keys() and addr in addr2income.keys():
            if addr2Intxnum[addr] == 0:
                addr2avgincome[addr] = 0
            else:
                addr2avgincome[addr] = addr2income[addr] / addr2Intxnum[addr]
    for addr in addrlist:
        if addr in addr2OutTxnum.keys() and addr in addr2outcome.keys():
            if addr2OutTxnum[addr] == 0:
                addr2avgincome[addr] = 0
            else:
                addr2avgincome[addr] = addr2outcome[addr] / addr2OutTxnum[addr]
    with open('normalItxAvgIncome.txt', 'w') as f:
        print(addr2avgincome, file=f)
    with open('normalItxAvgOutcome.txt', 'w') as f:
        print(addr2avgoutcome, file=f)
def norNtxAvgIOcome1():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normalIncomeNtx1.txt','r',encoding='utf-8') as f:
        addr2income = literal_eval(f.read())
    with open('normalOutcomeNtx1.txt','r',encoding='utf-8') as f:
        addr2outcome = literal_eval(f.read())
    with open('nor2numNIntx1.txt','r') as f:
        addr2Intxnum = literal_eval(f.read())
    with open('nor2numNOuttx1.txt','r',encoding='utf-8') as f:
        addr2OutTxnum = literal_eval(f.read())
    addr2avgincome = {}
    addr2avgoutcome = {}
    for addr in addrlist:
        if addr in addr2Intxnum.keys() and addr in addr2income.keys():
            if addr2Intxnum[addr] == 0:
                addr2avgincome[addr] = 0
            else:
                addr2avgincome[addr] = addr2income[addr] / addr2Intxnum[addr]
    for addr in addrlist:
        if addr in addr2OutTxnum.keys() and addr in addr2outcome.keys():
            if addr2OutTxnum[addr] == 0:
                addr2avgoutcome[addr] = 0
            else:
                addr2avgoutcome[addr] = addr2outcome[addr] / addr2OutTxnum[addr]
    with open('normalNtxAvgIncome1.txt', 'w') as f:
        print(addr2avgincome, file=f)
    with open('normalNtxAvgOutcome1.txt', 'w') as f:
        print(addr2avgoutcome, file=f)
def norNtxAvgIOcome2():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normalIncomeNtx2.txt','r',encoding='utf-8') as f:
        addr2income = literal_eval(f.read())
    with open('normalOutcomeNtx2.txt','r',encoding='utf-8') as f:
        addr2outcome = literal_eval(f.read())
    with open('nor2numNIntx2.txt','r') as f:
        addr2Intxnum = literal_eval(f.read())
    with open('nor2numNOuttx2.txt','r',encoding='utf-8') as f:
        addr2OutTxnum = literal_eval(f.read())
    addr2avgincome = {}
    addr2avgoutcome = {}
    for addr in addrlist:
        if addr in addr2Intxnum.keys() and addr in addr2income.keys():
            if addr2Intxnum[addr] == 0:
                addr2avgincome[addr] = 0
            else:
                addr2avgincome[addr] = addr2income[addr] / addr2Intxnum[addr]
    for addr in addrlist:
        if addr in addr2OutTxnum.keys() and addr in addr2outcome.keys():
            if addr2OutTxnum[addr] == 0:
                addr2avgincome[addr] = 0
            else:
                addr2avgoutcome[addr] = addr2outcome[addr] / addr2OutTxnum[addr]
    with open('normalNtxAvgIncome2.txt', 'w') as f:
        print(addr2avgincome, file=f)
    with open('normalNtxAvgOutcome2.txt', 'w') as f:
        print(addr2avgoutcome, file=f)
def norNtxAvgIOcome3():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normalIncomeNtx3.txt','r',encoding='utf-8') as f:
        addr2income = literal_eval(f.read())
    with open('normalOutcomeNtx3.txt','r',encoding='utf-8') as f:
        addr2outcome = literal_eval(f.read())
    with open('nor2numNIntx3.txt','r') as f:
        addr2Intxnum = literal_eval(f.read())
    with open('nor2numNOuttx3.txt','r',encoding='utf-8') as f:
        addr2OutTxnum = literal_eval(f.read())
    addr2avgincome = {}
    addr2avgoutcome = {}
    for addr in addrlist:
        if addr in addr2Intxnum.keys() and addr in addr2income.keys():
            if addr2Intxnum[addr] == 0:
                addr2avgincome[addr] = 0
            else:
                addr2avgincome[addr] = addr2income[addr] / addr2Intxnum[addr]
    for addr in addrlist:
        if addr in addr2OutTxnum.keys() and addr in addr2outcome.keys():
            if addr2OutTxnum[addr] == 0:
                addr2avgoutcome[addr] = 0
            else:
                addr2avgoutcome[addr] = addr2outcome[addr] / addr2OutTxnum[addr]
    with open('normalNtxAvgIncome3.txt', 'w') as f:
        print(addr2avgincome, file=f)
    with open('normalNtxAvgOutcome3.txt', 'w') as f:
        print(addr2avgoutcome, file=f)
def norNtxAvgIOcome4():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normalIncomeNtx4.txt','r',encoding='utf-8') as f:
        addr2income = literal_eval(f.read())
    with open('normalOutcomeNtx4.txt','r',encoding='utf-8') as f:
        addr2outcome = literal_eval(f.read())
    with open('nor2numNIntx4.txt','r') as f:
        addr2Intxnum = literal_eval(f.read())
    with open('nor2numNOuttx2.txt','r',encoding='utf-8') as f:
        addr2OutTxnum = literal_eval(f.read())
    addr2avgincome = {}
    addr2avgoutcome = {}
    for addr in addrlist:
        if addr in addr2Intxnum.keys() and addr in addr2income.keys():
            if addr2Intxnum[addr] == 0:
                addr2avgincome[addr] = 0
            else:
                addr2avgincome[addr] = addr2income[addr] / addr2Intxnum[addr]
    for addr in addrlist:
        if addr in addr2OutTxnum.keys() and addr in addr2outcome.keys():
            if addr2OutTxnum[addr] == 0:
                addr2avgoutcome[addr] = 0
            else:
                addr2avgoutcome[addr] = addr2outcome[addr] / addr2OutTxnum[addr]
    with open('normalNtxAvgIncome4.txt', 'w') as f:
        print(addr2avgincome, file=f)
    with open('normalNtxAvgOutcome4.txt', 'w') as f:
        print(addr2avgoutcome, file=f)
def norNtxAvgIOcome5():
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normalIncomeNtx5.txt','r',encoding='utf-8') as f:
        addr2income = literal_eval(f.read())
    with open('normalOutcomeNtx5.txt','r',encoding='utf-8') as f:
        addr2outcome = literal_eval(f.read())
    with open('nor2numNIntx5.txt','r') as f:
        addr2Intxnum = literal_eval(f.read())
    with open('nor2numNOuttx5.txt','r',encoding='utf-8') as f:
        addr2OutTxnum = literal_eval(f.read())
    addr2avgincome = {}
    addr2avgoutcome = {}
    for addr in addrlist:
        if addr in addr2Intxnum.keys() and addr in addr2income.keys():
            if addr2Intxnum[addr] == 0:
                addr2avgincome[addr] = 0
            else:
                addr2avgincome[addr] = addr2income[addr] / addr2Intxnum[addr]
    for addr in addrlist:
        if addr in addr2OutTxnum.keys() and addr in addr2outcome.keys():
            if addr2OutTxnum[addr] == 0:
                addr2avgoutcome[addr] = 0
            else:
                addr2avgoutcome[addr] = addr2outcome[addr] / addr2OutTxnum[addr]
    with open('normalNtxAvgIncome5.txt', 'w') as f:
        print(addr2avgincome, file=f)
    with open('normalNtxAvgOutcome5.txt', 'w') as f:
        print(addr2avgoutcome, file=f)
def mixnormalNtxAvgIOcome():
    # with open('normalNtxAvgIncome1.txt', 'r') as f:
    #     addr2avgincome1 = literal_eval(f.read())
    # with open('normalNtxAvgIncome2.txt', 'r') as f:
    #     addr2avgincome2 = literal_eval(f.read())
    # with open('normalNtxAvgIncome3.txt', 'r') as f:
    #     addr2avgincome3 = literal_eval(f.read())
    # with open('normalNtxAvgIncome4.txt', 'r') as f:
    #     addr2avgincome4 = literal_eval(f.read())
    # with open('normalNtxAvgIncome5.txt', 'r') as f:
    #     addr2avgincome5 = literal_eval(f.read())
    # with open('normalNtxAvgOutcome1.txt', 'r') as f:
    #     addr2avgoutcome1 = literal_eval(f.read())
    # with open('normalNtxAvgOutcome2.txt', 'r') as f:
    #     addr2avgoutcome2 = literal_eval(f.read())
    # with open('normalNtxAvgOutcome3.txt', 'r') as f:
    #     addr2avgoutcome3 = literal_eval(f.read())
    # with open('normalNtxAvgOutcome4.txt', 'r') as f:
    #     addr2avgoutcome4 = literal_eval(f.read())
    # with open('normalNtxAvgOutcome5.txt', 'r') as f:
    #     addr2avgoutcome5 = literal_eval(f.read())
    # with open('normalAddr.txt','r',encoding='utf-8') as f:
    #     addrlist = literal_eval(f.read())
    # addr2avgincome = {}
    # addr2avgoutcome = {}
    # for addr in addrlist:
    #     addr2avgincome[addr] = 0
    #     addr2avgoutcome[addr] = 0
    # for addr in addrlist:#将每个文件里面的平均值相加得到总的平均值
    #     if addr in addr2avgincome1.keys():
    #         addr2avgincome[addr] = addr2avgincome[addr] + addr2avgincome1[addr]
    #     if addr in addr2avgincome2.keys():
    #         addr2avgincome[addr] = addr2avgincome[addr] + addr2avgincome2[addr]
    #     if addr in addr2avgincome3.keys():
    #         addr2avgincome[addr] = addr2avgincome[addr] + addr2avgincome3[addr]
    #     if addr in addr2avgincome4.keys():
    #         addr2avgincome[addr] = addr2avgincome[addr] + addr2avgincome4[addr]
    #     if addr in addr2avgincome5.keys():
    #         addr2avgincome[addr] = addr2avgincome[addr] + addr2avgincome5[addr]
    #
    #     if addr in addr2avgoutcome1.keys():
    #         addr2avgoutcome[addr] = addr2avgoutcome[addr] + addr2avgoutcome1[addr]
    #     if addr in addr2avgoutcome2.keys():
    #         addr2avgoutcome[addr] = addr2avgoutcome[addr] + addr2avgoutcome2[addr]
    #     if addr in addr2avgoutcome3.keys():
    #         addr2avgoutcome[addr] = addr2avgoutcome[addr] + addr2avgoutcome3[addr]
    #     if addr in addr2avgoutcome4.keys():
    #         addr2avgoutcome[addr] = addr2avgoutcome[addr] + addr2avgoutcome4[addr]
    #     if addr in addr2avgoutcome5.keys():
    #         addr2avgoutcome[addr] = addr2avgoutcome[addr] + addr2avgoutcome5[addr]
    # with open('normalNtxAvgIncome.txt','w') as f:
    #     print(addr2avgincome,file=f)
    # with open('normalNtxAvgOutcome.txt','w') as f:
    #     print(addr2avgoutcome,file=f)
    #
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normalIncomeNtx1.txt','r',encoding='utf-8') as f:
        addr2income1 = literal_eval(f.read())
    with open('normalOutcomeNtx1.txt','r',encoding='utf-8') as f:
        addr2outcome1 = literal_eval(f.read())
    with open('nor2numNIntx1.txt','r') as f:
        addr2InTxnum1 = literal_eval(f.read())
    with open('nor2numNOuttx1.txt','r',encoding='utf-8') as f:
        addr2OutTxnum1 = literal_eval(f.read())

    with open('normalIncomeNtx2.txt','r',encoding='utf-8') as f:
        addr2income2 = literal_eval(f.read())
    with open('normalOutcomeNtx2.txt','r',encoding='utf-8') as f:
        addr2outcome2 = literal_eval(f.read())
    with open('nor2numNIntx2.txt','r') as f:
        addr2InTxnum2 = literal_eval(f.read())
    with open('nor2numNOuttx2.txt','r',encoding='utf-8') as f:
        addr2OutTxnum2 = literal_eval(f.read())

    with open('normalIncomeNtx3.txt','r',encoding='utf-8') as f:
        addr2income3 = literal_eval(f.read())
    with open('normalOutcomeNtx3.txt','r',encoding='utf-8') as f:
        addr2outcome3 = literal_eval(f.read())
    with open('nor2numNIntx3.txt','r') as f:
        addr2InTxnum3 = literal_eval(f.read())
    with open('nor2numNOuttx3.txt','r',encoding='utf-8') as f:
        addr2OutTxnum3 = literal_eval(f.read())

    with open('normalIncomeNtx4.txt','r',encoding='utf-8') as f:
        addr2income4 = literal_eval(f.read())
    with open('normalOutcomeNtx4.txt','r',encoding='utf-8') as f:
        addr2outcome4 = literal_eval(f.read())
    with open('nor2numNIntx4.txt','r') as f:
        addr2InTxnum4 = literal_eval(f.read())
    with open('nor2numNOuttx4.txt','r',encoding='utf-8') as f:
        addr2OutTxnum4 = literal_eval(f.read())

    with open('normalIncomeNtx5.txt','r',encoding='utf-8') as f:
        addr2income5 = literal_eval(f.read())
    with open('normalOutcomeNtx5.txt','r',encoding='utf-8') as f:
        addr2outcome5 = literal_eval(f.read())
    with open('nor2numNIntx5.txt','r') as f:
        addr2InTxnum5 = literal_eval(f.read())
    with open('nor2numNOuttx5.txt','r',encoding='utf-8') as f:
        addr2OutTxnum5 = literal_eval(f.read())
    addr2income = {}
    addr2outcome = {}
    addr2InTxnum = {}
    addr2OutTxnum = {}
    addr2avgincome = {}
    addr2avgoutcome = {}
    for addr in addrlist:
        addr2income[addr] = 0
        addr2outcome[addr] = 0
        addr2InTxnum[addr] = 0
        addr2OutTxnum[addr] = 0
        addr2avgincome[addr] = 0
        addr2avgoutcome[addr] = 0
    for addr in addrlist:
        if addr in addr2income1.keys():
            addr2income[addr] += addr2income1[addr]
        if addr in addr2income2.keys():
            addr2income[addr] += addr2income2[addr]
        if addr in addr2income3.keys():
            addr2income[addr] += addr2income3[addr]
        if addr in addr2income4.keys():
            addr2income[addr] += addr2income4[addr]
        if addr in addr2income5.keys():
            addr2income[addr] += addr2income5[addr]

        if addr in addr2outcome1.keys():
            addr2outcome[addr] += addr2outcome1[addr]
        if addr in addr2outcome2.keys():
            addr2outcome[addr] += addr2outcome2[addr]
        if addr in addr2outcome3.keys():
            addr2outcome[addr] += addr2outcome3[addr]
        if addr in addr2outcome4.keys():
            addr2outcome[addr] += addr2outcome4[addr]
        if addr in addr2outcome5.keys():
            addr2outcome[addr] += addr2outcome5[addr]

        if addr in addr2InTxnum1.keys():
            addr2InTxnum[addr] += addr2InTxnum1[addr]
        if addr in addr2InTxnum2.keys():
            addr2InTxnum[addr] += addr2InTxnum2[addr]
        if addr in addr2InTxnum3.keys():
            addr2InTxnum[addr] += addr2InTxnum3[addr]
        if addr in addr2InTxnum4.keys():
            addr2InTxnum[addr] += addr2InTxnum4[addr]
        if addr in addr2InTxnum5.keys():
            addr2InTxnum[addr] += addr2InTxnum[addr]

        if addr in addr2OutTxnum1.keys():
            addr2OutTxnum[addr] += addr2OutTxnum1[addr]
        if addr in addr2OutTxnum2.keys():
            addr2OutTxnum[addr] += addr2OutTxnum2[addr]
        if addr in addr2OutTxnum3.keys():
            addr2OutTxnum[addr] += addr2OutTxnum3[addr]
        if addr in addr2OutTxnum4.keys():
            addr2OutTxnum[addr] += addr2OutTxnum4[addr]
        if addr in addr2OutTxnum5.keys():
            addr2OutTxnum[addr] += addr2OutTxnum5[addr]
    with open('normalIncomeNtx.txt','w') as f:
        print(addr2income,file=f)
    with open('normalOutcomeNtx.txt','w') as f:
        print(addr2outcome,file=f)
    with open('nor2numNIntx.txt','w') as f:
        print(addr2InTxnum,file=f)
    with open('nor2numNOuttx.txt','w') as f:
        print(addr2OutTxnum,file=f)
    return
    for addr in addrlist:
        if addr2InTxnum[addr] != 0:
            addr2avgincome[addr] = addr2income[addr] / addr2InTxnum[addr]
        if addr2OutTxnum[addr] != 0:
            addr2outcome[addr] = addr2outcome[addr] / addr2OutTxnum[addr]
    print(addr2avgincome)



def mixnormalAvgIOcome():
    with open('normalIncomeNtx.txt','r') as f:
        addr2income1 = literal_eval(f.read())
    with open('normalOutcomeNtx.txt','r') as f:
        addr2outcome1 = literal_eval(f.read())
    with open('nor2numNIntx.txt','r') as f:
        addr2intx1 = literal_eval(f.read())
    with open('nor2numNOuttx.txt','r') as f:
        addr2outtx1 = literal_eval(f.read())
    with open('normalIncomeItx1.txt','r') as f:
        addr2income2 = literal_eval(f.read())
    with open('normalOutcomeItx1.txt','r') as f:
        addr2outcome2 = literal_eval(f.read())
    with open('nor2numIIntx1.txt','r') as f:
        addr2intx2 = literal_eval(f.read())
    with open('nor2numIOuttx.txt','r') as f:
        addr2outtx2 = literal_eval(f.read())
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addr2income = {}
    addr2outcome = {}
    addr2InTxnum = {}
    addr2OutTxnum = {}
    addr2avgincome = {}
    addr2avgoutcome = {}
    for addr in addrlist:
        addr2income[addr] = 0
        addr2outcome[addr] = 0
        addr2InTxnum[addr] = 0
        addr2OutTxnum[addr] = 0
        addr2avgincome[addr] = 0
        addr2avgoutcome[addr] = 0
    for addr in addrlist:
        if addr in addr2income1.keys():
            addr2income[addr] += addr2income1[addr]
        if addr in addr2income2.keys():
            addr2income[addr] += addr2income2[addr]
        if addr in addr2outcome1.keys():
            addr2outcome[addr] += addr2outcome1[addr]
        if addr in addr2outcome2.keys():
            addr2outcome[addr] += addr2outcome2[addr]
        if addr in addr2intx1.keys():
            addr2InTxnum[addr] += addr2intx1[addr]
        if addr in addr2intx2.keys():
            addr2InTxnum[addr] += addr2intx2[addr]
        if addr in addr2outtx1.keys():
            addr2OutTxnum[addr] += addr2outtx1[addr]
        if addr in addr2outtx2.keys():
            addr2OutTxnum[addr] += addr2outtx2[addr]
    # for addr in addrlist:
    #     if addr2InTxnum[addr] != 0:
    #         addr2avgincome[addr] = addr2income[addr] / addr2InTxnum[addr]
    #     if addr2OutTxnum[addr] != 0:
    #         addr2avgoutcome[addr] = addr2outcome[addr] / addr2OutTxnum[addr]
    # print(addr2avgincome)
    # print(addr2avgoutcome)
    with open('normalIncome.txt','w') as f:
        print(addr2income,file=f)
    with open('normalOutcome.txt','w') as f:
        print(addr2outcome,file=f)
    return
    with open('normalAvgIncome.txt','w') as f:
        print(addr2avgincome,file=f)
    with open('normalAvgOutcome.txt','w') as f:
        print(addr2avgoutcome,file=f)

    #如何计算每个地址的平均income和outcome，取加权平均值


def mixAvg():
    with open('scamAvgIncome.txt','r') as f:
        addr2avgincome1 = literal_eval(f.read())
    with open('scamAvgOutcome.txt','r') as f:
        addr2avgoutcome1 = literal_eval(f.read())
    with open('normalAvgIncome.txt','r') as f:
        addr2avgincome2 = literal_eval(f.read())
    with open('normalAvgOutcome.txt','r') as f:
        addr2avgoutcome2 = literal_eval(f.read())
    income2addrnum1 = {}
    income2addrnum2 = {}
    outcome2addrnum1 = {}
    outcome2addrnum2 = {}
    for addr,avgincome in addr2avgincome1.items():
        income2addrnum1[avgincome] = income2addrnum1.get(avgincome,0) + 1
    for addr,avgincome in addr2avgincome2.items():
        income2addrnum2[avgincome] = income2addrnum2.get(avgincome,0) + 1
    for addr,avgoutcome in addr2avgoutcome1.items():
        outcome2addrnum1[avgoutcome] = outcome2addrnum1.get(avgoutcome,0) + 1
    for addr,avgoutcome in addr2avgoutcome2.items():
        outcome2addrnum2[avgoutcome] = outcome2addrnum2.get(avgoutcome,0) + 1

    fig, ax = plt.subplots()

    zipped1 = zip(income2addrnum1.keys(), income2addrnum1.values())
    sort_zipped1 = sorted(zipped1, key=lambda x: (x[0]))
    result1 = zip(*sort_zipped1)
    x1, y1 = [list(x) for x in result1]
    cum = numpy.cumsum(y1)
    percentage1 = cum / list(cum)[-1]

    zipped2 = zip(outcome2addrnum1.keys(), outcome2addrnum1.values())
    sort_zipped2 = sorted(zipped2, key=lambda x: (x[0]))
    result2 = zip(*sort_zipped2)
    x2, y2 = [list(x) for x in result2]
    cum2 = numpy.cumsum(y2)
    percentage2 = cum2 / list(cum2)[-1]

    zipped3 = zip(income2addrnum2.keys(), income2addrnum2.values())
    sort_zipped3 = sorted(zipped3, key=lambda x: (x[0]))
    result3 = zip(*sort_zipped3)
    x3, y3 = [list(x) for x in result3]
    cum3 = numpy.cumsum(y3)
    percentage3 = cum3 / list(cum3)[-1]

    zipped4 = zip(outcome2addrnum2.keys(), outcome2addrnum2.values())
    sort_zipped4 = sorted(zipped4, key=lambda x: (x[0]))
    result4 = zip(*sort_zipped4)
    x4, y4 = [list(x) for x in result4]
    cum4 = numpy.cumsum(y4)
    percentage4 = cum4 / list(cum4)[-1]

    line1 = ax.plot(x1, percentage1, label='Avg Income Of scam addresses', color='black')
    line2 = ax.plot(x2, percentage2, label='Avg Outcome Of scam addresses', linestyle="--", color='black')
    line3 = ax.plot(x3, percentage3, label='Avg Income Of normal addresses', color='red')
    line4 = ax.plot(x4, percentage4, label='Avg Outcome Of normal addresses', linestyle="--", color='gray')
    ax.set_xlabel('Average Income/Outcome of Addresses (ETH)')
    ax.set_ylabel('CDF of Addresses')
    plt.xscale('log')
    ax.set_xlim(1)
    plt.legend()
    plt.savefig('avgIOcome.jpg')
    plt.show()
def mixfig6():
    with open('scamNtxliving.txt','r') as f:
        addr2living1 = literal_eval(f.read())
    with open('scamItxliving.txt','r') as f:
        addr2living2 = literal_eval(f.read())
    with open('scamEtxliving.txt','r') as f:
        addr2living3 = literal_eval(f.read())
    x = ['time<6h','6h≤time<12h','12h≤time<18h','18h≤time<24h','24h≤time<48h','48h≤time<1 week','1 week≤time<1 month','time>1 month']
    y1 = [0,0,0,0,0,0,0,0]
    y2 = [0,0,0,0,0,0,0,0]
    y3 = [0,0,0,0,0,0,0,0]
    index = np.arange(len(x))
    width = 0.2
    for addr,living in addr2living1.items():
        if living < 6:
            y1[0] += 1
        if 6 <= living and living <12:
            y1[1] += 1
        if 12 <= living and living < 18:
            y1[2] += 1
        if 18 <= living and living < 24:
            y1[3] += 1
        if 24 <= living and living < 48:
            y1[4] += 1
        if 48 <= living and living < 168:
            y1[5] += 1
        if 168 <= living and living < 720:
            y1[6] += 1
        if 720 <= living:
            y1[7] += 1
    for addr,living in addr2living2.items():
        if living < 6:
            y2[0] += 1
        if 6 <= living and living <12:
            y2[1] += 1
        if 12 <= living and living < 18:
            y2[2] += 1
        if 18 <= living and living < 24:
            y2[3] += 1
        if 24 <= living and living < 48:
            y2[4] += 1
        if 48 <= living and living < 168:
            y2[5] += 1
        if 168 <= living and living < 720:
            y2[6] += 1
        if 720 <= living:
            y2[7] += 1
    for addr,living in addr2living3.items():
        if living < 6:
            y3[0] += 1
        if 6 <= living and living <12:
            y3[1] += 1
        if 12 <= living and living < 18:
            y3[2] += 1
        if 18 <= living and living < 24:
            y3[3] += 1
        if 24 <= living and living < 48:
            y3[4] += 1
        if 48 <= living and living < 168:
            y3[5] += 1
        if 168 <= living and living < 720:
            y3[6] += 1
        if 720 <= living:
            y3[7] += 1
    plt.figure(figsize=(6, 6))
    plt.bar(x, y1, width, label="normal tx", color=['0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1'])
    plt.bar(index+width, y2, width, label="internal tx", color=['0.5','0.5','0.5','0.5','0.5','0.5','0.5','0.5'])
    plt.bar(index+width*2, y3, width, label="token tx", color=['0.9','0.9','0.9','0.9','0.9','0.9','0.9','0.9'])
    for a, b in zip(x, y1):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)
    x = index + width
    for a, b in zip(x, y2):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)
    x = index + width * 2
    for a, b in zip(x, y3):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)
    plt.xticks(rotation=30)
    ax = plt.gca()
    plt.yscale('log')
    ax.set_xlabel('living time of scam address')
    ax.set_ylabel('# of txs')
    plt.legend(loc='best')
    plt.savefig('fig6.jpg',bbox_inches = 'tight')
    plt.show()

def mixfig7():
    with open('scamNtxProfit.txt','r') as f:
        addr2prof1 = literal_eval(f.read())
    with open('scamItxProfit.txt','r') as f:
        addr2prof2 = literal_eval(f.read())
    profaxis = ['0-1k','1k-2k','2k-3k','3k-4k','4k-5k','5k-10k','10k-15k','>15k']
    prof2num1 = {}
    prof2num2 = {}
    for prof in profaxis:
        prof2num1[prof] = 0
        prof2num2[prof] = 0
    for addr,prof in addr2prof1.items():
        if prof / 1000 <= 1:
            prof2num1['0-1k'] += 1
        if prof / 1000 > 1 and prof / 1000 <= 2:
            prof2num1['1k-2k'] += 1
        if prof / 1000 > 2 and prof / 1000 <= 3:
            prof2num1['2k-3k'] += 1
        if prof / 1000 > 3 and prof / 1000 <= 4:
            prof2num1['3k-4k'] += 1
        if prof / 1000 > 4 and prof / 1000 <= 5:
            prof2num1['4k-5k'] += 1
        if prof / 1000 > 5 and prof / 1000 < 10:
            prof2num1['5k-10k'] += 1
        if prof / 1000 > 10 and prof / 1000 < 15:
            prof2num1['10k-15k'] += 1
        if prof / 1000 > 15:
            prof2num1['>15k'] += 1
    for addr,prof in addr2prof2.items():
        if prof / 1000 <= 1:
            prof2num2['0-1k'] += 1
        if prof / 1000 > 1 and prof / 1000 <= 2:
            prof2num2['1k-2k'] += 1
        if prof / 1000 > 2 and prof / 1000 <= 3:
            prof2num2['2k-3k'] += 1
        if prof / 1000 > 3 and prof / 1000 <= 4:
            prof2num2['3k-4k'] += 1
        if prof / 1000 > 4 and prof / 1000 <= 5:
            prof2num2['4k-5k'] += 1
        if prof / 1000 > 5 and prof / 1000 < 10:
            prof2num2['5k-10k'] += 1
        if prof / 1000 > 10 and prof / 1000 < 15:
            prof2num2['10k-15k'] += 1
        if prof / 1000 > 15:
            prof2num2['>15k'] += 1
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    index = numpy.arange(len(profaxis))
    width = 0.2
    for prof in profaxis:
        x1.append(prof)
        y1.append(prof2num1[prof])
    for prof in profaxis:
        x2.append(prof)
        y2.append(prof2num2[prof])
    plt.bar(profaxis,y1,width, label="normal tx")
    plt.bar(index+width,y2,width, label="internal tx")
    plt.xticks(rotation=30)
    plt.yscale('log')
    plt.legend()
    plt.title('scam address profit')
    plt.savefig('fig7.jpg')
    plt.show()

def mixfig9Ntx():
    with open('scamAddrNFig9Intx.txt','r',encoding='utf-8') as f:
        txnum2addrnum1 = literal_eval(f.read())
    with open('norAddrNFig9Intxs.txt','r',encoding='utf-8') as f:
        txnum2addrnum2 = literal_eval(f.read())
    with open('scamAddrNFig9Outtx.txt','r',encoding='utf-8') as f:
        txnum2addrnum3 = literal_eval(f.read())
    with open('norAddrNFig9Outtxs.txt','r',encoding='utf-8') as f:
        txnum2addrnum4 = literal_eval(f.read())
    zipped1 = zip(txnum2addrnum1.keys(), txnum2addrnum1.values())
    sort_zipped1 = sorted(zipped1, key=lambda x: (x[0]))
    result1 = zip(*sort_zipped1)
    x1, y1 = [list(x) for x in result1]
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y1)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x1,percentage,label='in txs of normal txs of scam address', color='black')

    zipped2 = zip(txnum2addrnum2.keys(), txnum2addrnum2.values())
    sort_zipped2 = sorted(zipped2, key=lambda x: (x[0]))
    result2 = zip(*sort_zipped2)
    x2, y2 = [list(x) for x in result2]
    cum = numpy.cumsum(y2)
    percentage2 = cum / list(cum)[-1]
    line2 = ax.plot(x2,percentage2,label='in txs of normal txs of normal address', linestyle="--", color='black')

    zipped3 = zip(txnum2addrnum3.keys(), txnum2addrnum3.values())
    sort_zipped3 = sorted(zipped3, key=lambda x: (x[0]))
    result3 = zip(*sort_zipped3)
    x3, y3 = [list(x) for x in result3]
    cum = numpy.cumsum(y3)
    percentage3 = cum / list(cum)[-1]
    line3 = ax.plot(x3, percentage3, label='out txs of normal txs of scam address', color='gray')

    zipped4 = zip(txnum2addrnum4.keys(), txnum2addrnum4.values())
    sort_zipped4 = sorted(zipped4, key=lambda x: (x[0]))
    result4 = zip(*sort_zipped4)
    x4, y4 = [list(x) for x in result4]
    cum = numpy.cumsum(y4)
    percentage4 = cum / list(cum)[-1]
    line4 = ax.plot(x4,percentage4,label='out txs of normal txs of normal address', linestyle="--", color='gray')
    ax.set_xlim(1)
    plt.xscale('log')
    plt.legend()
    ax.set_ylabel('CDF of Addresses')
    plt.title('# fo transactions')
    plt.savefig('Ntxfig9.jpg')
    plt.show()

def mixfig9Itx():
    with open('scamAddrIFig9Intx.txt','r',encoding='utf-8') as f:
        txnum2addrnum1 = literal_eval(f.read())
    with open('norAddrIFig9Intxs.txt','r',encoding='utf-8') as f:
        txnum2addrnum2 = literal_eval(f.read())
    with open('scamAddrIFig9Outtx.txt','r',encoding='utf-8') as f:
        txnum2addrnum3 = literal_eval(f.read())
    with open('norAddrIFig9Outtxs.txt','r',encoding='utf-8') as f:
        txnum2addrnum4 = literal_eval(f.read())
    zipped1 = zip(txnum2addrnum1.keys(), txnum2addrnum1.values())
    sort_zipped1 = sorted(zipped1, key=lambda x: (x[0]))
    result1 = zip(*sort_zipped1)
    x1, y1 = [list(x) for x in result1]
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y1)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x1,percentage,label='in txs of internal txs of scam address', color='black')
    zipped2 = zip(txnum2addrnum2.keys(), txnum2addrnum2.values())
    sort_zipped2 = sorted(zipped2, key=lambda x: (x[0]))
    result2 = zip(*sort_zipped2)
    x2, y2 = [list(x) for x in result2]
    cum = numpy.cumsum(y2)
    percentage = cum / list(cum)[-1]
    line2 = ax.plot(x2,percentage,label='in txs of internal txs of normal address', linestyle="--", color='black')
    zipped3 = zip(txnum2addrnum3.keys(), txnum2addrnum3.values())
    sort_zipped3 = sorted(zipped3, key=lambda x: (x[0]))
    result3 = zip(*sort_zipped3)
    x3, y3 = [list(x) for x in result3]
    cum = numpy.cumsum(y3)
    percentage3 = cum / list(cum)[-1]
    line3 = ax.plot(x3, percentage3, label='out txs of internal txs of scam address', color='gray')

    zipped4 = zip(txnum2addrnum4.keys(), txnum2addrnum4.values())
    sort_zipped4 = sorted(zipped4, key=lambda x: (x[0]))
    result4 = zip(*sort_zipped4)
    x4, y4 = [list(x) for x in result4]
    cum = numpy.cumsum(y4)
    percentage4 = cum / list(cum)[-1]
    line4 = ax.plot(x4, percentage4, label='out txs of internal txs of normal address', linestyle="--", color='gray')
    ax.set_xlim(1)
    plt.xscale('log')
    plt.legend()
    ax.set_ylabel('CDF of Addresses')
    plt.title('# of transactions')
    plt.savefig('Itxfig9.jpg')
    plt.show()

def mixfig9Etx():
    with open('scamAddrEFig9Intx.txt','r',encoding='utf-8') as f:
        txnum2addrnum1 = literal_eval(f.read())
    with open('norAddrEFig9Intxs.txt','r',encoding='utf-8') as f:
        txnum2addrnum2 = literal_eval(f.read())
    with open('scamAddrEFig9Outtx.txt','r',encoding='utf-8') as f:
        txnum2addrnum3 = literal_eval(f.read())
    with open('norAddrEFig9Outtx.txt','r',encoding='utf-8') as f:
        txnum2addrnum4 = literal_eval(f.read())
    zipped1 = zip(txnum2addrnum1.keys(), txnum2addrnum1.values())
    sort_zipped1 = sorted(zipped1, key=lambda x: (x[0]))
    result1 = zip(*sort_zipped1)
    x1, y1 = [list(x) for x in result1]
    cum = numpy.cumsum(y1)
    percentage1 = cum / list(cum)[-1]

    zipped2 = zip(txnum2addrnum2.keys(), txnum2addrnum2.values())
    sort_zipped2 = sorted(zipped2, key=lambda x: (x[0]))
    result2 = zip(*sort_zipped2)
    x2, y2 = [list(x) for x in result2]
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y2)
    percentage2 = cum / list(cum)[-1]

    zipped3 = zip(txnum2addrnum3.keys(), txnum2addrnum3.values())
    sort_zipped3 = sorted(zipped3, key=lambda x: (x[0]))
    result3 = zip(*sort_zipped3)
    x3, y3 = [list(x) for x in result3]
    cum = numpy.cumsum(y3)
    percentage3 = cum / list(cum)[-1]

    zipped4 = zip(txnum2addrnum4.keys(), txnum2addrnum4.values())
    sort_zipped4 = sorted(zipped4, key=lambda x: (x[0]))
    result4 = zip(*sort_zipped4)
    x4, y4 = [list(x) for x in result4]
    cum = numpy.cumsum(y4)
    percentage4 = cum / list(cum)[-1]

    line1 = ax.plot(x1,percentage1,label='in txs of token txs of scam address', color='black')
    line2 = ax.plot(x2,percentage2,label='in txs of token txs of normal address', linestyle="--", color='black')
    line3 = ax.plot(x3,percentage3,label='out txs of token txs of scam address', linestyle="--", color='gray')
    line4 = ax.plot(x4,percentage4,label='out txs of token txs of normal address', linestyle="--", color='gray')
    ax.set_xlim(1)
    # ax.set_xlabel('Average Income/Outcome of Addresses (ETH)')
    # ax.set_ylabel('CDF of Addresses')
    plt.xscale('log')
    plt.legend()
    plt.savefig('Etxfig9.jpg')
    plt.show()

def mixfig9Livingtime():
    # with open('scam2athirdLivingtime3.txt','r') as f:#都是时间戳的差
    #     scamaddr2living3 = literal_eval(f.read())
    # with open('normal2athirdtime3.txt','r') as f:
    #     noraddr2living3 = literal_eval(f.read())
    # for addr,timestamp in scamaddr2living3.items():
    #     scamaddr2living3[addr] = timestamp / 86400
    # for addr,timestamp in noraddr2living3.items():
    #     noraddr2living3[addr] = timestamp / 86400#计算天数，进行cdf计算，先统计天数和地址数的关系
    # scamday2addrnum = {}
    # norday2addrnum = {}
    # for addr,daynum in scamaddr2living3.items():
    #     scamday2addrnum[daynum] = scamday2addrnum.get(daynum,0) + 1
    # for addr,daynum in noraddr2living3.items():
    #     norday2addrnum[daynum] = norday2addrnum.get(daynum,0) + 1
    # with open('scamday2addrnum.txt','w') as f:
    #     print(scamday2addrnum,file=f)
    # with open('norday2addrnum.txt','w') as f:
    #     print(norday2addrnum,file=f)

    with open('scamday2addrnum.txt','r') as f:
        scamday2addrnum = literal_eval(f.read())
    with open('norday2addrnum.txt','r') as f:
        norday2addrnum = literal_eval(f.read())

    zipped = zip(scamday2addrnum.keys(), scamday2addrnum.values())
    sort_zipped1 = sorted(zipped, key=lambda x: (x[0]))
    result1 = zip(*sort_zipped1)
    x1, y1 = [list(x) for x in result1]
    cum = numpy.cumsum(y1)
    percentage1 = cum / list(cum)[-1]

    zipped = zip(norday2addrnum.keys(), norday2addrnum.values())
    sort_zipped2 = sorted(zipped, key=lambda x: (x[0]))
    result2 = zip(*sort_zipped2)
    x2, y2 = [list(x) for x in result2]
    cum = numpy.cumsum(y2)
    percentage2 = cum / list(cum)[-1]
    fig, ax = plt.subplots()
    line1 = ax.plot(x1,percentage1,label='Living Time of Scam Addresses',ls='--')
    line2 = ax.plot(x2,percentage2,label='Living Time of Normal Addresses',ls='-')

    x_major_locator = MultipleLocator(250)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.set_ylabel('CDF of Addresses')
    ax.set_xlabel('the living time of address(day)')
    ax.set_xlim(1)
    plt.legend()
    plt.savefig('fig9living.jpg')
    plt.show()

def mixfig9livcon():#livingtime的数量分布，应该可以画柱状图然后取log
    with open('scamday2addrnum.txt','r') as f:
        scamday2addrnum = literal_eval(f.read())
    with open('norday2addrnum.txt','r') as f:
        norday2addrnum = literal_eval(f.read())
    # print(len(norday2addrnum))
    x = ['time<1d','1d≤time<2d','2d≤time<3d','3d≤time<4d','4d≤time<7d','7d≤time<30d','30d≤time<60d','>60d']
    y1 = [0,0,0,0,0,0,0,0]
    y2 = [0,0,0,0,0,0,0,0]
    index = np.arange(len(x))
    for day,addrnum in scamday2addrnum.items():
        if day < 1:
            y1[0] += scamday2addrnum[day]
        if day >= 1 and day < 2:
            y1[1] += scamday2addrnum[day]
        if day >= 2 and day < 3:
            y1[2] += scamday2addrnum[day]
        if day >= 3 and day < 4:
            y1[3] += scamday2addrnum[day]
        if day >= 4 and day < 7:
            y1[4] += scamday2addrnum[day]
        if day >= 7 and day < 30:
            y1[5] += scamday2addrnum[day]
        if day >= 30 and day < 60:
            y1[6] += scamday2addrnum[day]
        if day > 60:
            y1[7] += scamday2addrnum[day]
    for day,addrnum in norday2addrnum.items():
        if day < 1:
            y2[0] += norday2addrnum[day]
        if day >= 1 and day < 2:
            y2[1] += norday2addrnum[day]
        if day >= 2 and day < 3:
            y2[2] += norday2addrnum[day]
        if day >= 3 and day < 4:
            y2[3] += norday2addrnum[day]
        if day >= 4 and day < 7:
            y2[4] += norday2addrnum[day]
        if day >= 7 and day < 30:
            y2[5] += norday2addrnum[day]
        if day >= 30 and day < 60:
            y2[6] += norday2addrnum[day]
        if day > 60:
            y2[7] += norday2addrnum[day]
    width = 0.4
    plt.figure(figsize=(6, 6.5))
    plt.bar(x, y1, width, label="scam address")
    plt.bar(index + width, y2, width, label="normal address")
    for a, b in zip(x, y1):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)
    x = index + width
    for a, b in zip(x, y2):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)
    plt.xticks(rotation=30)
    ax = plt.gca()
    plt.yscale('log')
    ax.set_xlabel('living time(d) of scam/normal address')
    plt.legend(loc='best')
    plt.savefig('fig9livconc.jpg', bbox_inches='tight')
    plt.show()

def graph1():
    df = pandas.read_csv('exp_group_victim.csv')
    temp_list = df['componentnumber']
    temp_dict = {}
    for each in temp_list:
        if each not in temp_dict.keys():
            temp_dict[each] = 1
        else:
            temp_dict[each] = temp_dict[each] + 1
    no_list = []
    number_list = []
    for k, v in temp_dict.items():
        no_list.append(k)
        number_list.append(v)
    df = pandas.DataFrame({'id': no_list, '#': number_list})
    df.to_csv(r'number of each group victim')

def scamNtxMFG():
    with open('addr.txt', 'r') as f:
        addrlist = literal_eval(f.read())
    df1 = pandas.read_csv('ntx.csv')
    for index, row in df1.iterrows():
        if isinstance(row['from'],str):
            row['from'] = row['from'].lower()
        if isinstance(row['to'],str):
            row['to'] = row['to'].lower()
    dict1 = {}
    addrlist2 = []
    for addr in addrlist:
        dict1[addr] = {}
    for index, row in df1.iterrows():
        # if row['from'] in addrlist and row['to'] in addrlist:
        if row['from'] in addrlist:
            dict1[row['from']][row['to']] = dict1[row['from']].get(row['to'],0) + float(row['value'])
    with open('scamNtxMFG.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['from','to','value'])
        for fromaddr,to2value in dict1.items():
            for toaddr,value in dict1[fromaddr].items():
                writer.writerow([fromaddr,toaddr,value])
                addrlist2.append(toaddr)
            addrlist2.append(fromaddr)
    addrlist2 = list(set(addrlist2))
    print(addrlist2)
    solp = []
    for addr in addrlist:
        if addr not in addrlist2:
            solp.append(addr)
    with open('scamNtxSolP.txt','w',encoding='utf-8') as f:
        print(solp,file=f)
    #独立点，和没有被识别为欺诈地址的其他地址进行交易
def scamItxMFG():
    with open('addr.txt', 'r') as f:
        addrlist = literal_eval(f.read())
    df1 = pandas.read_csv('itx.csv')
    for index, row in df1.iterrows():
        if isinstance(row['from'],str):
            row['from'] = row['from'].lower()
        if isinstance(row['to'],str):
            row['to'] = row['to'].lower()
    dict1 = {}
    addrlist2 = []
    for addr in addrlist:
        dict1[addr] = {}
    for index, row in df1.iterrows():
        # if row['from'] in addrlist and row['to'] in addrlist:
        if row['from'] in addrlist:
            dict1[row['from']][row['to']] = dict1[row['from']].get(row['to'],0) + float(row['value'])#value的单位是wei
    with open('scamItxMFG.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['from','to','value'])
        for fromaddr,to2value in dict1.items():
            for toaddr,value in dict1[fromaddr].items():
                writer.writerow([fromaddr,toaddr,value])
                addrlist2.append(toaddr)
            addrlist2.append(fromaddr)
    # print(addrlist2)
    addrlist2 = list(set(addrlist2))
    print(len(addrlist2))
    solp = []
    for addr in addrlist:
        if addr not in addrlist2:
            solp.append(addr)
    with open('scamItxSolP.txt','w',encoding='utf-8') as f:
        print(solp,file=f)
def scamccg():#合约创建图，data必须是0
    df1 = pandas.read_csv('ntx.csv')
    df2 = pandas.read_csv('itx.csv')
    ccg = {}#key是创建者，value是被创建的合约
    for index, row in df1.iterrows():
        if not isinstance(row['to'],str):#如果to是null则表示创建一个地址为contract address的合约
            ccg[row['from']] = row['contractAddress']
    for index, row in df2.iterrows():
        if not isinstance(row['to'],str):#如果to是null则表示创建一个地址为contract address的合约
            ccg[row['from']] = row['contractAddress']
    with open('scamccg.txt','w') as f:
        print(ccg,file=f)

def norccgn():#不能通过eoa得到ccg，无法得到创建者
    df1 = pandas.read_csv('normalAddrntx2.csv')
    ccg = {}
    for index, row in df1.iterrows():
        if not isinstance(row['to'],str):#如果to是null则表示创建一个地址为contract address的合约
            ccg[row['from']] = row['contractAddress']
    with open('normalccgn.txt','w') as f:
        print(ccg,file=f)

def norccgi():#不能通过eoa得到ccg，无法得到创建者
    df1 = pandas.read_csv('normalAddritx2.csv')
    ccg = {}
    for index, row in df1.iterrows():
        if not isinstance(row['to'],str):#如果to是null则表示创建一个地址为contract address的合约
            ccg[row['from']] = row['contractAddress']
    with open('normalccgi.txt','w') as f:
        print(ccg,file=f)

def eoaOrcontractScam():#欺诈节点是eoa还是合约
    #获取所有的地址，是eoa则为1
    with open('addr.txt','r') as f:
        scamaddr = literal_eval(f.read())
    addrlist = scamaddr
    session = requests.Session()
    addr2eoa = {}
    for addr in addrlist:
        addr2eoa[addr] = 0
        url = "https://api.etherscan.io/api?module=contract&action=getabi&address="+ addr + "&apikey=" + apikey
        message = literal_eval(session.get(url).text)['message']
        if message == "NOTOK":
            addr2eoa[addr] = 1
        elif message == "OK":
            addr2eoa[addr] = 0
    with open('addr2eoa1.txt','w') as f:
        print(addr2eoa,file=f)
# def eoaOrcontractNormal1():
#     with open('addr2eoa1.txt','r') as f:
#         addr2eoa1 = literal_eval(f.read())
#     addrlist = list(addr2eoa1.keys())
#     with open('addr.txt','r') as f:
#         scamaddr = literal_eval(f.read())
#     addr2eoa = {}
#     for addr in scamaddr:
#         addr2eoa[addr] = 0
#         if addr in addrlist:
#             addr2eoa[addr] = 1
#         print(addr2eoa[addr])
#     with open('addr2eoa11.txt', 'w') as f:
#         print(addr2eoa, file=f)
def eoaOrcontract2():#正常节点是eoa还是合约
    #获取所有的地址，是eoa则为1
    with open('normalAddr.txt','r') as f:
        noraddr = literal_eval(f.read())
    addrlist = noraddr
    session = requests.Session()
    addr2eoa = {}
    for addr in addrlist:
        addr2eoa[addr] = 0
        url = "https://api.etherscan.io/api?module=contract&action=getabi&address="+ addr + "&apikey=" + apikey
        message = literal_eval(session.get(url).text)['message']
        if message == "NOTOK":
            addr2eoa[addr] = 1
        elif message == "OK":
            addr2eoa[addr] = 0
    with open('addr2eoa2.txt','w') as f:
        print(addr2eoa,file=f)
def scamcig():#统计调用对象是合约的次数，交易地址to是合约地址
    df1 = pandas.read_csv('ntx.csv')
    df2 = pandas.read_csv('itx.csv')
    with open('addr.txt','r') as f:
        scamaddrlist = literal_eval(f.read())
    with open('addr2eoa1.txt','r') as f:
        addr2eoa = literal_eval(f.read())
    cig = {}#是否应该区分欺诈地址和正常地址，如果区分则会漏掉一部分to地址是正常地址的交易
    for index, row in df1.iterrows():
        if isinstance(row['to'], str) and row['to'] in scamaddrlist:#有可能交易接收方不是欺诈地址里面的合约地址或者eoa地址，所以要保证接收方是欺诈地址
            iseoa = addr2eoa[row['to']]#如果是eoa则不是调用，如果不是eoa则是合约地址可以调用，需要统计到cig
            if iseoa == 0:
                cig[(row['from'],row['to'])] = cig.get((row['from'],row['to']),0) + 1

    with open('scamcig.txt','w') as f:
        print(cig,file=f)

def norcign():
    df1 = pandas.read_csv('normalAddrntx2.csv')
    with open('addr2eoa2.txt','r') as f:
        addr2eoa = literal_eval(f.read())
    cig = {}#交易接收方是合约地址，则将交易发送方和接收方的边的权重计算出来，如果某条边没有被提及则记为0权重，先初始化所有地址之间的交易权重为0
    #应该是稀疏矩阵，还是说只存储权重非0的边
    for index, row in df1.iterrows():
        if isinstance(row['to'], str) and addr2eoa[row['to']] == 0:#有可能接收方既不是合约地址也不是eoa
            iseoa = addr2eoa[row['to']]  # 如果接收方是eoa则不是调用，如果不是eoa则是合约地址可以调用，需要统计到cig
            if iseoa == 0:
                cig[(row['from'], row['to'])] = cig.get((row['from'], row['to']), 0) + 1
    with open('normalcig.txt', 'w') as f:
        print(cig, file=f)

def norcigi():
    df1 = pandas.read_csv('normalAddritx2.csv')
    with open('addr2eoa2.txt','r') as f:
        addr2eoa = literal_eval(f.read())
    cig = {}  # 如果是eoa则不是调用，如果不是eoa则是合约地址可以调用，需要统计到cig
    for index, row in df1.iterrows():
        if isinstance(row['to'], str):
            iseoa = addr2eoa[row['to']]
            if iseoa == 0:
                cig[(row['from'], row['to'])] = cig.get((row['from'], row['to']), 0) + 1
    with open('normalcig.txt', 'w') as f:
        print(cig, file=f)
def mycig():
    with open('addr2eoa1.txt','r') as f:
        addr2eoa1 = literal_eval(f.read())
    with open('addr2eoa2.txt','r') as f:
        addr2eoa2 = literal_eval(f.read())
    addr2eoa = addr2eoa1.update(addr2eoa2)
    with open('addr.txt','r') as f:
        scamaddrlist = literal_eval(f.read())
    with open('normalAddr.txt','r') as f:#需要注意正常地址也是有限的数据集，所以仍要排除没有采集到的地址数据导致的异常
        noraddrlist = literal_eval(f.read())
    addrlist = scamaddrlist + noraddrlist
    df1 = pandas.read_csv('ntx.csv')
    df2 = pandas.read_csv('itx.csv')
    df3 = pandas.read_csv('normalAddrntx2.csv')
    df4 = pandas.read_csv('normalAddritx2.csv')
    cig = {}
    for index, row in df1.iterrows():
        cig[(row['from'], row['to'])] = cig.get((row['from'], row['to']), 0)#初始化边
        if isinstance(row['to'], str) and row['to'] in addrlist:
            iseoa = addr2eoa1[row['to']]
            if iseoa == 0:
                cig[(row['from'], row['to'])] = cig.get((row['from'], row['to']), 0) + 1
    for index, row in df2.iterrows():
        cig[(row['from'], row['to'])] = cig.get((row['from'], row['to']), 0)#初始化边
        if isinstance(row['to'], str) and row['to'] in addrlist:
            iseoa = addr2eoa1[row['to']]
            if iseoa == 0:
                cig[(row['from'], row['to'])] = cig.get((row['from'], row['to']), 0) + 1
    for index, row in df3.iterrows():
        cig[(row['from'], row['to'])] = cig.get((row['from'], row['to']), 0)#初始化边
        if isinstance(row['to'], str) and row['to'] in addrlist:
            iseoa = addr2eoa1[row['to']]
            if iseoa == 0:
                cig[(row['from'], row['to'])] = cig.get((row['from'], row['to']), 0) + 1
    for index, row in df4.iterrows():
        cig[(row['from'], row['to'])] = cig.get((row['from'], row['to']), 0)#初始化边
        if isinstance(row['to'], str) and row['to'] in addrlist:
            iseoa = addr2eoa1[row['to']]
            if iseoa == 0:
                cig[(row['from'], row['to'])] = cig.get((row['from'], row['to']), 0) + 1
    with open('cig.txt', 'w') as f:
        print(cig, file=f)
#每个文件的含义
# exp_add_profit.csv 地址和利润
# exp_group_id.csv 地址和聚类后的块号
# exp_group_victim.csv 地址和聚类后的？
# extend_group_id.csv 地址和聚类后的块号
# extend_group_profit.csv 利润和块号
# final_type2_all_trans_new.csv ????
# extend_group_start_time.csv 开始时间和块号
# final_type2_in_trans_USD.csv 地址和利润
# final_type2_trans_USD.csv地址和利润
# need_to_reget.csv 地址
# node.csv 无法打开
# number of each group victim.csv 每个块的受害者
# scam group NO.csv 地址和块号
# scam group time and profit.csv 欺诈地址和时间和利润
# tx_into.csv ?
def character13():
    with open('scamIncome.txt','r',encoding='utf-8') as f:
        scamIncome = literal_eval(f.read())
    with open('scamOutcome.txt','r',encoding='utf-8') as f:
        scamOutcome = literal_eval(f.read())
    with open('scamAvgIncome.txt', 'r', encoding='utf-8') as f:
        scamAvgIncome = literal_eval(f.read())
    with open('scamAvgOutcome.txt', 'r', encoding='utf-8') as f:
        scamAvgOutcome = literal_eval(f.read())
    with open('scamInTx.txt','r') as f:#总的in tx num，但是命名不一致
        scamInTx = literal_eval(f.read())
    with open('scamOutTx.txt','r') as f:
        scamOutTx = literal_eval(f.read())
    with open('scam2athirdLivingtime3.txt', 'r') as f:
        scamLivingtime = literal_eval(f.read())
    with open('scam2athirdIntxnum1.txt','r') as f:
        scam2athirdIntxnum1 = literal_eval(f.read())
    with open('scam2athirdIntxnum2.txt','r') as f:
        scam2athirdIntxnum2 = literal_eval(f.read())
    with open('scam2athirdIntxnum3.txt','r') as f:
        scam2athirdIntxnum3 = literal_eval(f.read())
    with open('scam2athirdOuttxnum1.txt','r') as f:
        scam2athirdOuttxnum1 = literal_eval(f.read())
    with open('scam2athirdOuttxnum2.txt','r') as f:
        scam2athirdOuttxnum2 = literal_eval(f.read())
    with open('scam2athirdOuttxnum3.txt','r') as f:
        scam2athirdOuttxnum3 = literal_eval(f.read())

    with open('normalIncome.txt','r') as f:
        normalIncome = literal_eval(f.read())
    with open('normalOutcome.txt','r') as f:
        normalOutcome = literal_eval(f.read())
    with open('normalAvgIncome.txt','r') as f:
        normalAvgIncome = literal_eval(f.read())
    with open('normalAvgOutcome.txt','r') as f:
        normalAvgOutcome = literal_eval(f.read())
    with open('normal2athirdIntxnum.txt','r') as f:#其实是intx num
        normal2athirdIntxnum = literal_eval(f.read())
    with open('normal2athirdOuttxnum.txt','r') as f:
        normal2athirdOuttxnum = literal_eval(f.read())
    with open('normal2athirdtime3.txt', 'r') as f:
        normalLivingtime = literal_eval(f.read())
    with open('normal2athirdIntxnum1.txt','r') as f:
        normal2athirdIntxnum1 = literal_eval(f.read())
    with open('normal2athirdIntxnum2.txt','r') as f:
        normal2athirdIntxnum2 = literal_eval(f.read())
    with open('normal2athirdIntxnum3.txt','r') as f:
        normal2athirdIntxnum3 = literal_eval(f.read())
    with open('normal2athirdOuttxnum1.txt','r') as f:
        normal2athirdOuttxnum1 = literal_eval(f.read())
    with open('normal2athirdOuttxnum2.txt','r') as f:
        normal2athirdOuttxnum2 = literal_eval(f.read())
    with open('normal2athirdOuttxnum3.txt','r') as f:
        normal2athirdOuttxnum3 = literal_eval(f.read())
    with open('addr.txt', 'r', encoding='utf-8') as f:
        scamaddrlist = literal_eval(f.read())
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        noraddrlist = literal_eval(f.read())
    scamlist = []
    norlist = []
    for addr in scamaddrlist:
        scamdict = {}
        scamdict['address'] = addr
        scamdict['allIncome'] = scamIncome.get(addr,0)
        scamdict['allOutcome'] = scamOutcome.get(addr,0)
        scamdict['avgIncome'] = scamAvgIncome.get(addr,0)
        scamdict['avgOutcome'] = scamAvgOutcome.get(addr,0)
        scamdict['intxs'] = scamInTx.get(addr,0)
        scamdict['outtxs'] = scamOutTx.get(addr,0)
        scamdict['livingTime'] = scamLivingtime.get(addr,0)
        scamdict['front1/3in'] = scam2athirdIntxnum1.get(addr,0)
        scamdict['middle1/3in'] = scam2athirdIntxnum2.get(addr,0)
        scamdict['last1/3in'] = scam2athirdIntxnum3.get(addr,0)
        scamdict['front1/3out'] = scam2athirdOuttxnum1.get(addr,0)
        scamdict['middle1/3out'] = scam2athirdOuttxnum2.get(addr,0)
        scamdict['last1/3out'] = scam2athirdOuttxnum3.get(addr,0)
        scamdict['type'] = 1
        scamlist.append(scamdict)
    for addr in noraddrlist:
        nordict = {}
        nordict['address'] = addr
        nordict['allIncome'] = normalIncome.get(addr,0)
        nordict['allOutcome'] = normalOutcome.get(addr,0)
        nordict['avgIncome'] = normalAvgIncome.get(addr,0)
        nordict['avgOutcome'] = normalAvgOutcome.get(addr, 0)
        nordict['intxs'] = normal2athirdIntxnum.get(addr, 0)
        nordict['outtxs'] = normal2athirdOuttxnum.get(addr, 0)
        nordict['livingTime'] = normalLivingtime.get(addr,0)
        nordict['front1/3in'] = normal2athirdIntxnum1.get(addr, 0)
        nordict['middle1/3in'] = normal2athirdIntxnum2.get(addr, 0)
        nordict['last1/3in'] = normal2athirdIntxnum3.get(addr, 0)
        nordict['front1/3out'] = normal2athirdOuttxnum1.get(addr, 0)
        nordict['middle1/3out'] = normal2athirdOuttxnum2.get(addr, 0)
        nordict['last1/3out'] = normal2athirdOuttxnum3.get(addr, 0)
        nordict['type'] = 0
        norlist.append(nordict)
    with open('mlchar.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["address","allIncome","allOutcome","avgIncome","avgOutcome","intxs","outtxs","livingTime","front1/3in","middle1/3in","last1/3in","front1/3out","middle1/3out","last1/3out","type"])
        for scamdict in scamlist:
            if int(scamdict['intxs']) != 0 and int(scamdict['outtxs']) != 0:
                writer.writerow([scamdict['address'], scamdict['allIncome'], scamdict['allOutcome'], scamdict['avgIncome'], scamdict['avgOutcome'], scamdict['intxs'],
                                 scamdict['outtxs'], scamdict['livingTime'], scamdict['front1/3in'], scamdict['middle1/3in'], scamdict['last1/3in'], scamdict['front1/3out'], scamdict['middle1/3out'],
                                 scamdict['last1/3out'], scamdict['type']])
        for nordict in norlist:
            if int(nordict['intxs']) != 0 and int(nordict['outtxs']) != 0:
                writer.writerow([nordict['address'], nordict['allIncome'], nordict['allOutcome'], nordict['avgIncome'],
                                 nordict['avgOutcome'], nordict['intxs'],nordict['outtxs'], nordict['livingTime'], nordict['front1/3in'], nordict['middle1/3in'], nordict['last1/3in'],
                                 nordict['front1/3out'], nordict['middle1/3out'], nordict['last1/3out'], nordict['type']])


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split#sklearn没有训练、测试、验证集这么划分
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def logReg():#使用所有特征进行传统的机器学习
    data = pandas.read_csv('mlchar.csv')[["allIncome","allOutcome","avgIncome","avgOutcome","intxs","outtxs","livingTime","front1/3in","middle1/3in","last1/3in","front1/3out","middle1/3out","last1/3out","type"]]
    data = shuffle(data)
    X1 = data["allIncome"].to_list()
    X2 = data["allOutcome"].to_list()
    X3 = data["avgIncome"].to_list()
    X4 = data["avgOutcome"].to_list()
    X5 = data["intxs"].to_list()
    X6 = data["outtxs"].to_list()
    X7 = data["livingTime"].to_list()
    X8 = data["front1/3in"].to_list()
    X9 = data["middle1/3in"].to_list()
    X10 = data["last1/3in"].to_list()
    X11 = data["front1/3out"].to_list()
    X12 = data["middle1/3out"].to_list()
    X13 = data["last1/3out"].to_list()
    X = numpy.array([X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13]).T
    Y = data["type"].to_list()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)#3:1的训练测试比

    print('LogisticRegression')
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Ypred = logreg.predict(X_test)
    print(logreg.coef_)
    print(classification_report(Y_test, Ypred))
    # print( mean_squared_error(Y_test, Ypred))

    print('KNeighborsClassifier')
    logreg = KNeighborsClassifier(n_neighbors=3)
    logreg.fit(X_train, Y_train)
    Ypred = logreg.predict(X_test)
    # print(logreg.coef_)
    print(classification_report(Y_test, Ypred))
    # print( mean_squared_error(Y_test, Ypred))

    print('svm')
    logreg = svm.SVC()
    logreg.fit(X_train, Y_train)
    Ypred = logreg.predict(X_test)
    print(classification_report(Y_test, Ypred))
    # print( mean_squared_error(Y_test, Ypred))

    print('DecisionTreeClassifier')
    logreg = DecisionTreeClassifier(random_state=1,splitter="best")
    logreg.fit(X_train, Y_train)
    Ypred = logreg.predict(X_test)
    print(classification_report(Y_test, Ypred))
    # print(mean_squared_error(Y_test, Ypred))

    print('GaussianNB')#就是贝叶斯
    logreg = GaussianNB()
    logreg.fit(X_train, Y_train)
    Ypred = logreg.predict(X_test)
    print(classification_report(Y_test, Ypred))
    # print(mean_squared_error(Y_test, Ypred))

    print('RandomForestClassifier')
    logreg = RandomForestClassifier()
    logreg.fit(X_train, Y_train)
    Ypred = logreg.predict(X_test)
    print(classification_report(Y_test, Ypred))
    # print(mean_squared_error(Y_test, Ypred))

    print('GradientBoostingClassifier')
    logreg = GradientBoostingClassifier()
    logreg.fit(X_train, Y_train)
    Ypred = logreg.predict(X_test)
    print(classification_report(Y_test, Ypred))
    # print(mean_squared_error(Y_test, Ypred))

    print('MLPClassifier')#就是dnn
    logreg = MLPClassifier(verbose=True, max_iter=300)
    logreg.fit(X_train, Y_train)
    Ypred = logreg.predict(X_test)
    print(classification_report(Y_test, Ypred))
    # print(mean_squared_error(Y_test, Ypred))



#使用相邻的节点也是欺诈节点规则判断，这个不能直接用于图神经网络
def getscamNodeEdgeCSV():
    # with open('addr.txt','r',encoding='utf-8') as f:
    #     scamaddrlist = literal_eval(f.read())
    # with open('normalAddr.txt','r',encoding='utf-8') as f:
    #     noraddrlist = literal_eval(f.read())
    # df1 = pandas.read_csv('ntx.csv')#收集所有欺诈地址有关的交易，也就是欺诈地址的一跳地址内的unknown地址作为待验证的数据集
    # df2 = pandas.read_csv('itx.csv')
    # frames = [df1,df2]
    # df = pandas.concat(frames)
    # print(len(df))
    # nodecsv = {}
    # edgecsv = {}
    # for index, row in df.iterrows():
    #     if isinstance(row['from'],str):
    #         nodecsv[row['from']] = {}
    #         edgecsv[row['from']] = {}
    #     if isinstance(row['to'],str):
    #         nodecsv[row['to']] = {}
    #         edgecsv[row['to']] = {}
    # for index, row in df.iterrows():
    #     if isinstance(row['to'],str):
    #         addr = row['to']
    #         nodecsv[addr]['indegree'] = nodecsv[addr].get('indegree', 0) + 1#nodecsv是欺诈交易中涉及的每个节点的各个属性，包括出度入度类型
    #         if row['to'] in scamaddrlist:
    #             nodecsv[addr]['type'] = 'scam'
    #         elif row['to'] in noraddrlist:
    #             nodecsv[addr]['type'] = 'normal'
    #         elif row['to'] not in scamaddrlist and row['to'] not in noraddrlist:
    #             nodecsv[addr]['type'] = 'unknown'
    #     if isinstance(row['from'],str):
    #         addr = row['from']
    #         nodecsv[addr]['outdegree'] = nodecsv[addr].get('outdegree', 0) + 1
    #         if row['from'] in scamaddrlist:
    #             nodecsv[addr]['type'] = 'scam'
    #         elif row['from'] in noraddrlist:
    #             nodecsv[addr]['type'] = 'normal'
    #         elif row['from'] not in scamaddrlist and row['from'] not in noraddrlist:
    #             nodecsv[addr]['type'] = 'unknown'
    #     if isinstance(row['from'],str) and isinstance(row['to'],str):
    #         edgecsv[row['from']][row['to']] = edgecsv[row['from']].get(row['to'],0) + 1
    #     print(index)
    # with open('mlnode.txt','w',encoding='utf-8',newline='') as f:
    #     print(nodecsv,file=f)
    # with open('mledge.txt','w',encoding='utf-8',newline='') as f:
    #     print(edgecsv,file=f)

    with open('mlnode.txt', 'r', encoding='utf-8') as f:#统计欺诈节点相关交易涉及的所有节点的信息，命名有问题，不是用于机器学习的数据，因为机器学习还要统计正常节点的数据，用于识别欺诈团伙scam group的数据
        nodecsv = literal_eval(f.read())
    with open('mledge.txt', 'r', encoding='utf-8') as f:#统计欺诈节点相关交易涉及的所有交易边的权重
        edgecsv = literal_eval(f.read())
    for addr,attr in nodecsv.items():
        nodecsv[addr]['degree'] = nodecsv[addr].get('indegree',0) + nodecsv[addr].get('outdegree',0)
    with open('mlnode.csv','w',encoding='utf-8',newline='') as f:#
        writer = csv.writer(f)
        writer.writerow(['Id','type','indegree','outdegree','degree'])
        for addr,attr in nodecsv.items():
            writer.writerow([addr,nodecsv[addr]['type'],nodecsv[addr].get('indegree',0),nodecsv[addr].get('outdegree',0),nodecsv[addr]['degree']])
    with open('mledge.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Source','Target','Weight'])
        for fromaddr,toaddr2weight in edgecsv.items():
            for toaddr,weight in toaddr2weight.items():
                writer.writerow([fromaddr,toaddr,weight])
    return
def getnorIOAddr():#正常地址的一跳节点
    df1 = pandas.read_csv('normalAddrntx2.csv')
    df2 = pandas.read_csv('normalAddritx2.csv')
    frames = [df1, df2]
    df = pandas.concat(frames)
    addr2addr = {}
    for index, row in df.iterrows():
        if isinstance(row['from'], str):
            addr2addr[row['from']] = []
        if isinstance(row['to'], str):
            addr2addr[row['to']] = []
    for index, row in df.iterrows():
        if isinstance(row['from'], str) and isinstance(row['to'], str):
            addr2addr[row['from']].append(row['to'])
    # with open('normaladdr2addr.txt', 'w', encoding='utf-8') as f:
    #     print(addr2addr,file=f)
    with open('normaladdr2addr.csv', 'w', encoding='utf-8', newline='') as f:#正常节点交易的一跳的所有地址
        writer = csv.writer(f)
        writer.writerow(['Source','Target'])
        for addr,addrlist in addr2addr.items():
            for addr2 in addrlist:
                writer.writerow([addr,addr2])
from collections import defaultdict
def getnorNodeEdgeCSV():#正常节点及其一跳节点,mlnode2不是机器学习用的数据，是正常节及其一跳节点区分类型后的csv
    addr2addr = pandas.read_csv('normaladdr2addr.csv')
    with open('addr.txt', 'r', encoding='utf-8') as f:
        scamaddrlist = literal_eval(f.read())
    with open('normalAddr.txt', 'r', encoding='utf-8') as f:
        noraddrlist = literal_eval(f.read())
    nodecsv = defaultdict(defaultdict)#
    edgecsv = defaultdict(defaultdict)
    for index,row in addr2addr.iterrows():#区分正常节点及其一跳节点的类型
        nodecsv[row['Source']]['outdegree'] = nodecsv[row['Source']].setdefault('outdegree',0) + 1
        if row['Source'] in scamaddrlist:
            nodecsv[row['Source']]['type'] = nodecsv[row['Source']].setdefault('type','scam')
        if row['Source'] in noraddrlist:
            nodecsv[row['Source']]['type'] = nodecsv[row['Source']].setdefault('type','normal')
        if row['Source'] not in scamaddrlist and row['Source'] not in noraddrlist:
            nodecsv[row['Source']]['type'] = nodecsv[row['Source']].setdefault('type','unknown')

        nodecsv[row['Target']]['indegree'] = nodecsv[row['Target']].setdefault('indegree',0) + 1
        if row['Target'] in scamaddrlist:
            nodecsv[row['Target']]['type'] = nodecsv[row['Target']].setdefault('type','scam')
        if row['Target'] in noraddrlist:
            nodecsv[row['Target']]['type'] = nodecsv[row['Target']].setdefault('type','normal')
        if row['Target'] not in scamaddrlist and row['Target'] not in noraddrlist:
            nodecsv[row['Target']]['type'] = nodecsv[row['Target']].setdefault('type','unknown')
        edgecsv[row['Source']][row['Target']] = edgecsv[row['Source']].setdefault(row['Target'],0) + 1
    with open('mlnode2.txt', 'w', encoding='utf-8') as f:
        print(nodecsv, file=f)
    with open('mledge2.txt', 'w', encoding='utf-8') as f:
        print(edgecsv, file=f)
    for addr, attr in nodecsv.items():
        nodecsv[addr]['degree'] = nodecsv[addr].get('indegree', 0) + nodecsv[addr].get('outdegree', 0)
    with open('mlnode2.csv', 'w', encoding='utf-8', newline='') as f:#正常节及其一跳节点区分类型后的csv
        writer = csv.writer(f)
        writer.writerow(['Id', 'type', 'indegree', 'outdegree', 'degree'])
        for addr, attr in nodecsv.items():
            writer.writerow([addr,nodecsv[addr]['type'],nodecsv[addr].get('indegree',0),nodecsv[addr].get('outdegree',0),nodecsv[addr]['degree']])
    with open('mledge2.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Source', 'Target', 'Weight'])
        for fromaddr, toaddr2weight in edgecsv.items():
            for toaddr, weight in toaddr2weight.items():
                writer.writerow([fromaddr, toaddr, weight])

def mixNodeEdgeCsv():
    normaladdr2addr = pandas.read_csv('normaladdr2addr.csv')
    mlnode = pandas.read_csv('mlnode.csv')#欺诈节点及其一跳的节点
    mledge = pandas.read_csv('mledge.csv')#欺诈节点及其一跳的交易
    mlnode2 = pandas.read_csv('mlnode2.csv')
    mledge2 = pandas.read_csv('mledge2.csv')
    print(len(mlnode))
    print(len(mlnode2))
    print(len(mlnode)+len(mlnode2))
    newNorNode = mlnode2[(mlnode2['degree'] < 5)]#挑选度小于5的正常节点及其一跳节点
    newNorNode = newNorNode.sample(len(mlnode),random_state=3)#随机挑选和欺诈node及其一跳节点相同数量的正常node及其一跳节点，但还要算和这些正常node相连的node，根据这些正常node的地址，找直接连接的node
    newNodelist = list(newNorNode['Id'])#先随机选正常node及其一跳节点然后再找和其直接连接的，即需要迭代一次，二跳节点
    directNode = []
    for addr in newNodelist:
        if addr in normaladdr2addr.keys():#起点边
            directNode = directNode + normaladdr2addr[addr]#出的边
    for addr,addrlist in normaladdr2addr.items():#正常节点一跳的地址
        for newaddr in newNodelist:#随机的一跳的地址
            if newaddr in addrlist:
                directNode.append(addr)#入的边
    newNodelist = newNodelist + directNode
    newNorNode = mlnode2[(mlnode2['degree'] < 5) & mlnode2['Id'].isin(newNodelist)]#度<5的正常节点及其一跳节点
    newNorEdge = mledge2[(mledge2['Source'].isin(newNodelist)) & (mledge2['Target'].isin(newNodelist))]#从正常节点交易中筛选
    newNorNode.to_csv('mlnode5.csv')#中间node
    newNorEdge.to_csv('mledge5.csv')#中间edge
    frames1 = [mlnode,newNorNode]
    frames2 = [mledge,newNorEdge]#没有算正常节点及其一跳的节点
    nodecsv = pandas.concat(frames1).drop_duplicates()
    edgecsv = pandas.concat(frames2).drop_duplicates()
    print(len(nodecsv))
    print(len(edgecsv))
    nodecsv['Label'] = ""
    nodecsv['Interval'] = ""
    nodecsv.to_csv('node3.csv')
    edgecsv.to_csv('edge3.csv')#不应该用于图神经网络，只能用于欺诈集群

    # nodecsv = pandas.read_csv('node.csv')
    # edgecsv = pandas.read_csv('edge.csv')
    # print(len(nodecsv))
    # print(len(edgecsv))
def unknownNode():#1：2的欺诈一跳节点和正常一跳节点,按照数量差不多是1700和3400，按照训练集，测试集，验证集的比例应该是3：1：1，发现不需要提取未知节点
    mlnode = pandas.read_csv('mlnode.csv')  # 欺诈节点及其一跳的节点
    mledge = pandas.read_csv('mledge.csv')  # 欺诈节点及其一跳的交易
    mlnode2 = pandas.read_csv('mlnode2.csv') # 正常节点及其一跳的节点
    mledge2 = pandas.read_csv('mledge2.csv') # 正常节点及其一跳的交易
    mlnode3 = []# 未知节点一跳的节点
    mledge3 = []# 未知节点一跳的交易
    with open('mlnode3.csv', 'w', encoding='utf-8', newline='') as f:#欺诈节点的一跳未知节点
        writer = csv.writer(f)
        writer.writerow(['Id', 'type', 'indegree', 'outdegree', 'degree'])
        for index,row in mlnode.iterrows():
            if row['type'] == 'unknown':
                writer.writerow([row['Id'],row['type'],row['indegree'],row['outdegree'],row['degree']])

        for index,row in mlnode2.iterrows():#正常节点的一跳未知节点
            if row['type'] == 'unknown':
                writer.writerow([row['Id'],row['type'],row['indegree'],row['outdegree'],row['degree']])



#三次机器学习的训练样本数：
#node:332191 332063 332124
#edge:197687 197687 197689

def getpred():
    pred = pandas.read_csv('predict-gdc.csv')
    num = 0
    for index,row in pred.iterrows():
        if row['pred'] == 'tensor(1)':
            num += 1
    print(num)
#11249
def wrongfig11():
    # addr = '0x83053c32b7819f420dcfed2d218335fe430fe3b5'
    # addr = '0x21f74c6bbc1e3ab9f0205e12de3a9daa14351aed'
    # addr = '0x4fed1fc4144c223ae3c1553be203cdfcbd38c581'
    # addr = '0x4bf722014e54aeab05fcf1519e6e4c0c3f742e43'
    # addr = '0x536a6ba0d913d5d6a4ce2c6eb7ed0de3c0f0b89e'
    # addr = '0x74886dd45da313f3d3d32235fed4595827d0865d'#没有scam
    # addr = '0x917a417d938b9f9e6ae7f9e5253fb6de410343e3'#节点太少
    # # addr = '0x7c9001c50ea57c1b2ec1e3e63cf04c297534bfc1'
    # addr = '0xba83e9ce38b10522e3d6061a12779b7526839eda'#第二次迭代节点太多
    addr = '0x050022eba4d55d242d9568794961df5596f968f2'
    #选0x4fed1
    postfix = '0x05002'
    txtfile = 'specialNode' + postfix + '.txt'
    csvnode1 = 'specialNode' + postfix + '.csv'
    csvnode2 = 'specialNode2' + postfix + '.csv'
    csvedge1 = 'specialEdge' + postfix + '.csv'
    csvedge2 = 'specialEdge2' + postfix + '.csv'
    mledge = pandas.read_csv('mledge.csv')
    mlnode = pandas.read_csv('mlnode.csv')
    mledge2 = pandas.read_csv('mledge2.csv')
    mlnode2 = pandas.read_csv('mlnode2.csv')

    # specialNode = []
    # for index,row in mledge.iterrows():
    #     if isinstance(row['Source'],str) and row['Source'] == addr:
    #         specialNode.append(row['Target'])
    #     if isinstance(row['Target'],str) and row['Target'] == addr:
    #         specialNode.append(row['Source'])
    # for index,row in mledge2.iterrows():
    #     if isinstance(row['Source'],str) and row['Source'] == addr:
    #         specialNode.append(row['Target'])
    #     if isinstance(row['Target'],str) and row['Target'] == addr:
    #         specialNode.append(row['Source'])
    # with open('specialNode.txt','w') as f:#将和addr有关联的点放到specialNode中，同时取对应的边，没有向外拓展一次，所以中心一直都是addr
    #     print(specialNode,file=f)
    # # # return
    # #
    # specialEdge1 = mledge[(mledge['Source'].notnull()) & (mledge['Source'] == addr) | (mledge['Target'].notnull()) & (mledge['Target'] == addr)]
    # specialEdge2 = mledge2[(mledge2['Source'].notnull()) & (mledge2['Source'] == addr) | (mledge2['Target'].notnull()) & (mledge2['Target'] == addr)]
    # frames = [specialEdge1,specialEdge2]
    # specialEdge = pandas.concat(frames)
    # specialEdge.to_csv('specialEdge.csv')
    # # return
    with open(txtfile,'r') as f:
        specialNode = literal_eval(f.read())
    specialNodecsv1 = mlnode[(mlnode['Id'].isin(specialNode))]#单节点的直接连接节点的全部信息
    specialNodecsv2 = mlnode2[(mlnode2['Id'].isin(specialNode))]
    frames = [specialNodecsv1,specialNodecsv2]
    specialNodecsv = pandas.concat(frames)
    specialNodecsv.reset_index(drop=True,inplace=True)
    #

    specialEdge1 = pandas.read_csv(csvedge1)
    specialEdge2csv1 = mledge[(isinstance(mledge['Source'],str))  & (mledge['Source'].isin(specialNode)) | (mledge['Target'].notnull()) & (mledge['Target'].isin(specialNode)) ]
    specialEdge2csv2 = mledge2[(isinstance(mledge2['Source'],str))  & (mledge2['Source'].isin(specialNode)) | (mledge2['Target'].notnull()) & (mledge2['Target'].isin(specialNode)) ]
    frames = [specialEdge1,specialEdge2csv1,specialEdge2csv2]
    specialEdge2 = pandas.concat(frames)
    specialEdge2.to_csv(csvedge2)
    with open('norMinerAddr.txt','r') as f:#小写地址
        norMinerAddr = literal_eval(f.read())
    with open('norExchangeAddr.txt','r') as f:
        norExchangeAddr = literal_eval(f.read())
    with open('norTokenAddr.txt', 'r') as f:
        norTokenAddr = literal_eval(f.read())
    for index,row in specialNodecsv.iterrows():
        if row['Id'].lower() in norMinerAddr:
            row['type'] = 'Miner'
            specialNodecsv.iloc[index] = row
        if row['Id'].lower() in norExchangeAddr:
            row['type'] = 'Exchange'
            specialNodecsv.iloc[index] = row
        if row['Id'].lower() in norTokenAddr:
            row['type'] = 'Token'
            specialNodecsv.iloc[index] = row
    specialNodecsv.to_csv(csvnode1)
    # # return
    # #将所有的直接连接的节点添加为待添加边的节点，然后对其边进行统计
    # #再迭代一次

    with open(txtfile, 'r') as f:
        specialNode = literal_eval(f.read())
    with open('norMinerAddr.txt','r') as f:#小写地址
        norMinerAddr = literal_eval(f.read())
    with open('norExchangeAddr.txt','r') as f:
        norExchangeAddr = literal_eval(f.read())
    with open('norTokenAddr.txt', 'r') as f:
        norTokenAddr = literal_eval(f.read())

    specialEdge2 = pandas.read_csv(csvedge2)
    specialNode2list1 = list(specialEdge2['Source'])
    specialNode2list2 = list(specialEdge2['Target'])
    specialNode2 = list(set(specialNode2list1 + specialNode2list2 + specialNode))#specialNode2才是用来画fig11的集合

    specialNode2csv1 = mlnode[(mlnode['Id'].isin(specialNode2))]
    specialNode2csv2 = mlnode2[(mlnode2['Id'].isin(specialNode2))]
    frames = [specialNode2csv1,specialNode2csv2]
    specialNode2csv = pandas.concat(frames)
    specialNode2csv.to_csv(csvnode2)

    specialNode2csv = pandas.read_csv(csvnode2)
    specialNode2csv.reset_index(drop=True,inplace=True)
    for index,row in specialNode2csv.iterrows():
        if row['Id'].lower() in norMinerAddr:
            row['type'] = 'Miner'
            specialNode2csv.iloc[index] = row
        if row['Id'].lower() in norExchangeAddr:
            row['type'] = 'Exchange'
            specialNode2csv.iloc[index] = row
        if row['Id'].lower() in norTokenAddr:
            row['type'] = 'Token'
            specialNode2csv.iloc[index] = row
    specialNode2csv.to_csv(csvnode2)
    with open(txtfile,'w') as f:
        print(specialNode2,file=f)

def fig11():
    # mledge = pandas.read_csv('mledge.csv')
    # mlnode = pandas.read_csv('mlnode.csv')
    # mledge2 = pandas.read_csv('mledge2.csv')
    # mlnode2 = pandas.read_csv('mlnode2.csv')
    # with open('specialNode0x83053.txt','r') as f:
    #     specialNode = literal_eval(f.read())
    # specialNodecsv1 = mlnode[(mlnode['Id'].isin(specialNode))]#单节点的直接连接节点的全部信息
    # specialNodecsv2 = mlnode2[(mlnode2['Id'].isin(specialNode))]
    # frames = [specialNodecsv1,specialNodecsv2]
    # specialNodecsv = pandas.concat(frames)
    # specialNodecsv.reset_index(drop=True,inplace=True)
    # with open('norMinerAddr.txt', 'r') as f:  # 小写地址
    #     norMinerAddr = literal_eval(f.read())
    # with open('norExchangeAddr.txt', 'r') as f:
    #     norExchangeAddr = literal_eval(f.read())
    # with open('norTokenAddr.txt', 'r') as f:
    #     norTokenAddr = literal_eval(f.read())
    # for index, row in specialNodecsv.iterrows():
    #     if row['Id'].lower() in norMinerAddr:
    #         row['type'] = 'Miner'
    #         specialNodecsv.iloc[index] = row
    #     if row['Id'].lower() in norExchangeAddr:
    #         row['type'] = 'Exchange'
    #         specialNodecsv.iloc[index] = row
    #     if row['Id'].lower() in norTokenAddr:
    #         row['type'] = 'Token'
    #         specialNodecsv.iloc[index] = row
    # specialNodecsv.to_csv('specialNode0x83053.csv')

    # specialNode = pandas.read_csv('specialNode0x4fed1.csv')
    # specialEdge = pandas.read_csv('specialEdge0x4fed1.csv')#ba83e点太少
    #
    # specialNode = pandas.read_csv('specialNode0x83053.csv')
    # specialEdge = pandas.read_csv('specialEdge0x83053.csv')

    # postfix = '0x917a4'
    postfix = '0x7c900'#像叉子，没有被认可
    # postfix = '0x05002'
    csvnode2 = 'specialNode2' + postfix + '.csv'
    csvedge2 = 'specialEdge2' + postfix + '.csv'
    specialNode = pandas.read_csv(csvnode2)
    specialEdge = pandas.read_csv(csvedge2)

    # fig11node = pandas.read_csv(r'C:\Users\ljh\Desktop\fig11node.csv')
    # fig11edge = pandas.read_csv(r'C:\Users\ljh\Desktop\fig11edge.csv')

    with open(r'C:\Users\ljh\Desktop\fig11node.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id','type','indegree','outdegree','degree'])
        for index,row in specialNode.iterrows():
            writer.writerow([row['Id'],row['type'],row['indegree'],row['outdegree'],row['degree']])
    with open(r'C:\Users\ljh\Desktop\fig11edge.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Source','Target','Weight'])
        for index,row in specialEdge.iterrows():
            writer.writerow([row['Source'],row['Target'],row['Weight']])
    return
    ExchangeNodeList = []
    ExchangeEdgeList = []
    TokenNodeList = []
    TokenEdgeList = []
    ScamNodeList = []
    ScamEdgeList = []
    MinerNodeList = []
    MinerEdgeList = []
    UnknownNodeList = []
    UnknownEdgeList = []
    colorMap = []
    for index,row in specialNode.iterrows():
        if row['type'] == 'Exchange':
            ExchangeNodeList.append(row['Id'])
            colorMap.append('yellow')
        if row['type'] == 'Token':
            TokenNodeList.append(row['Id'])
            colorMap.append('green')
        if row['type'] == 'scam':
            ScamNodeList.append(row['Id'])
            colorMap.append('orange')
        if row['type'] == 'Miner':
            MinerNodeList.append(row['Id'])
            colorMap.append('pink')
        if row['type'] == 'unknown':
            UnknownNodeList.append(row['Id'])
            colorMap.append('red')

    specialNodeList = list(specialNode['Id'])
    G1 = nx.DiGraph()
    G1.add_nodes_from(specialNodeList)
    edges = []
    for index,row in specialEdge.iterrows():
        mytuple = (row['Source'],row['Target'],row['Weight'])
        edges.append(mytuple)
        if row['Source'] in ExchangeNodeList or row['Target'] in ExchangeNodeList:
            ExchangeEdgeList.append((row['Source'],row['Target'],))
        if row['Source'] in TokenNodeList or row['Target'] in TokenNodeList:
            TokenEdgeList.append((row['Source'],row['Target'],))
        if row['Source'] in ScamNodeList or row['Target'] in ScamNodeList:
            ScamEdgeList.append((row['Source'],row['Target'],))
        if row['Source'] in MinerNodeList or row['Target'] in MinerNodeList:
            MinerEdgeList.append((row['Source'],row['Target'],))
        if row['Source'] in UnknownNodeList or row['Target'] in UnknownNodeList:
            UnknownEdgeList.append((row['Source'],row['Target'],))
    G1.add_weighted_edges_from(edges)
    weightlist = specialEdge['Weight']
    testlist = ['0x5a8c64ce7d652a671fab9de1e11a08a72bae4389','0x20a41844d89c646ec2bee518e4b2376a883f9ab9','0x83053c32b7819f420dcfed2d218335fe430fe3b5']
    testlist2 = [addr for addr in specialNodeList if addr not in testlist ]
    # pos = nx.spring_layout(G1,threshold=1e-4,iterations=50,scale=3,seed=1,k=0.005)#kind=1
    # pos = nx.spring_layout(G1)#kind=2
    # pos = nx.spectral_layout(G1,weight=weightlist)
    # pos = nx.shell_layout(G1,center=(3,4))#不行，椭圆
    # pos = nx.kamada_kawai_layout(G1)
    # pos = nx.multipartite_layout(G1)
    # pos = nx.fruchterman_reingold_layout(G1)#kind=3
    pos = nx.fruchterman_reingold_layout(G1,k=0.001)#kind=4
    pos = nx.fruchterman_reingold_layout(G1,k=0.0001)#kind=5
    # pos = nx.fruchterman_reingold_layout(G1,scale=2)#kind=6，改点大小后kind=10
    # pos = nx.fruchterman_reingold_layout(G1,k=0.00001)#kind=7
    # pos = nx.fruchterman_reingold_layout(G1,k=0.0005)#kind=8
    pos = nx.fruchterman_reingold_layout(G1,k=0.005)#kind=9，改点大小后kind=11
    # pos = nx.spiral_layout(G1)#年轮形
    # pos = nx.planar_layout(G1)#
    # pos = nx.bipartite_layout(G1,nodes=testlist2)#不行，三角形
    # pos = nx.random_layout(G1)
    # pos = nx.rescale_layout(G1)
    # print(ExchangeEdgeList)
    plt.subplot(111)
    nx.draw_networkx_nodes(G1,pos,nodelist=ExchangeNodeList,node_color='yellow',node_size=4,label='Exchange')
    nx.draw_networkx_nodes(G1,pos,nodelist=TokenNodeList,node_color='green',node_size=4,label='Token')
    nx.draw_networkx_nodes(G1,pos,nodelist=ScamNodeList,node_color='red',node_size=4,label='Scam')
    nx.draw_networkx_nodes(G1,pos,nodelist=MinerNodeList,node_color='pink',node_size=4,label='Miner')
    nx.draw_networkx_nodes(G1,pos,nodelist=UnknownNodeList,node_color='#FAF0E6',node_size=4,label='Unknown')#kind=10

    nx.draw_networkx_edges(G1,pos,edgelist=ExchangeEdgeList,width=0.1,edge_color='yellow',arrowsize=1,arrowstyle='->',arrows=False)
    nx.draw_networkx_edges(G1,pos,edgelist=TokenEdgeList,width=0.1,edge_color='green',arrowsize=1,arrowstyle='->',arrows=False)
    nx.draw_networkx_edges(G1,pos,edgelist=ScamEdgeList,width=0.1,edge_color='red',arrowsize=1,arrowstyle='->',arrows=False)
    nx.draw_networkx_edges(G1,pos,edgelist=MinerEdgeList,width=0.1,edge_color='pink',arrowsize=1,arrowstyle='->',arrows=False)
    nx.draw_networkx_edges(G1,pos,edgelist=UnknownEdgeList,width=0.1,edge_color='#FAF0E6',arrowsize=1,arrowstyle='->',arrows=False)

    # G = nx.path_graph(4)
    # shells = [[0], [1, 2, 3]]
    # pos = nx.shell_layout(G, shells)
    # nx.draw_networkx(G)
    kind = 11
    mytitle = postfix + str(kind)
    plt.title(mytitle)
    plt.legend()
    plt.savefig('fig11.jpg')
    plt.show()

def taxoNormalAddr():
    normalAddr = pandas.read_csv('ethereum_tagged_address.csv',encoding='ISO-8859-1')
    minerAddr = []
    tokenAddr = []
    exchangeAddr = []
    for index,row in normalAddr.iterrows():
        if 'blacklist' not in row['label'].lower() and 'miner' in row['label'].lower():
            minerAddr.append(row['addr'].lower())
        elif 'blacklist' not in row['label'].lower() and 'exchange' in row['label'].lower():
            exchangeAddr.append(row['addr'].lower())
        elif 'blacklist' not in row['label'].lower():
            tokenAddr.append(row['addr'].lower())
    with open('norMinerAddr.txt','w') as f:
        print(minerAddr,file=f)
    with open('norExchangeAddr.txt','w') as f:
        print(exchangeAddr,file=f)
    with open('norTokenAddr.txt','w') as f:
        print(tokenAddr,file=f)
def fig12():
    mledge = pandas.read_csv('mledge.csv')
    mlnode = pandas.read_csv('mlnode.csv')
    # with open('addr.txt', 'r', encoding='utf-8') as f:
    #     scamaddrlist = literal_eval(f.read())
    mlnode = mlnode[(mlnode['degree'] > 0) & (mlnode['type']=='scam')]#
    degreeScamnode = list(mlnode['Id'])#只筛选度>0且互相连接的欺诈节点
    mledge = mledge[(mledge['Source'].notnull()) & (mledge['Source'].isin(degreeScamnode)) & (mledge['Target'].notnull()) & (mledge['Target'].isin(degreeScamnode))]#欺诈节点相互连接的边
    # mledge.to_csv('scamNodeDegreeEdge.csv')#欺诈边：两边都是欺诈节点的边
    mledgeSource = list(mledge['Source'])#欺诈边的起点
    mledgeTarget = list(mledge['Target'])#
    mledgelist = mledgeSource + mledgeTarget#欺诈边节点的列表
    mlnode = mlnode[(mlnode['Id'].isin(mledgelist))]#满足一定条件的欺诈节点
    # mlnode.to_csv('scamNodeDegree.csv')#和没有迭代没区别
    print(mlnode.shape[0])
    print(mledge.shape[0])

    #34.8+4.62+0.96+0.81+0.2=41.39 816个 0x83053c32b7819f420dcfed2d218335fe430fe3b5
    #2.03 40个
    #1.78 35个
    #1.57 31个
    #1.32 26个
    #0.81 16个
    #0.81 16个
    #0.71 14个
    #0.71 14个
    #0.66 13个
    #0.61 12个
    #0.56 11个
    #0.56 11个
    #0.56 11个
    #0x0111a93692b5ccff5c4b006d88d82664f76dfd21 2.03 40
    #0xb1898895a2ac9ef09afeda89ee559427b7142f2e 1.78 35
    #0x7f183cce61efcabb8d2eb1340f2a94c5eb1bf8e5 1.57 31
    #0x408fac2d2d3d3ac4e356aa188119f5624fd8c9f6 1.32 26
    #0x13098fdefba2129b8873bcff5c9ce2de1cf7f32f 0.81 16
    #0x24ecad110119e2c763f1afc7f4e5dd159238d143 0.81 16
    #0xb70f65c1faa2f1160d006b585ebf564781e8dae6 0.71 14
    #0xe7d60131d3773a8cdf75162c2cb8be459600d9b2 0.71 14
    #0x2b065809f6ec6df32878bcd26711a0e2bcf59c26 0.66 14
    #0x3da2bb8c828643d13c399e71d9c7b70852d45492 0.61 13
    #0xa9d499f4e14d5046cbdfd01cde3e8c3ea0ac94d7 0.61 12
    #0xe8ddafab7212f988d84c973471c6feab851769e8 0.56 11
    #0x0084515449b037205a33d6d3940a5684126aa4b5 0.56 11
    #0xb7686c343be859d7875f8d6bb7c228858e229b8d 0.51 11
    #0x5416400b0d5798dfdcefa2d22335f18982ee22fa 0.46 10
    #0x1ea3ecc937bae79353d198db2cdab1e5cbe625dd 0.46 10
    #0xe0430890f9c7010ad6743293fc352680e2f01b4e 0.46 10
    #0xdafc2a2d4e070f863cfcb5887bc2516e05e18877 0.46 10
    #0xe7860fd151cbbad28141dfd4cc7cd6d090e4ead8 0.46 10
    #0x7970780b07b6a8c814eaeab01a09c4f922906da4 0.41 9
    #0xb3958757fd6d2dfbcc934f39faa1f82210da6f50 0.41 9
    #0xf9c8c57fe9eaabc5dfef8a8d03ef308f2a28c91c 0.41 9
    #0x7fa4cbcfed0362516ab52c16d4fde35a72a5c829 0.36 8
    #0x1ee6b25a0e3f855a10916766df3a056e924e5dce 0.36 8
    #0xbbdef8b12babd3ab6018566bc17ec3aa302b8348 0.36 8
    #0.21 5
    #48个community点数大于5 占比12%，通过在fig12中的scam拓展unknown节点获得新的节点和边，然后再次模块化得到新的community
    #397个community
def fig12x():
    mlnode = pandas.read_csv('mlnode.csv')
    mledge = pandas.read_csv('mledge.csv')
    scamNodeDegree = pandas.read_csv('scamNodeDegree.csv')
    scamNodeDegreeList = list(scamNodeDegree['Id'])
    scamNodeDegreeEdge = pandas.read_csv('scamNodeDegreeEdge.csv')
    G1 = nx.DiGraph()
    G1.add_nodes_from(scamNodeDegreeList)
    edges = []
    for index,row in scamNodeDegreeEdge.iterrows():
        mytuple = (row['Source'],row['Target'],row['Weight'])
        edges.append(mytuple)
    G1.add_weighted_edges_from(edges)
    pos = nx.fruchterman_reingold_layout(G1)
    nx.draw_networkx_edges(G1,pos,arrows=False)
    # nx.draw_networkx(G1,pos)
    nx.draw(G1, with_labels=True, font_weight='bold')
    for c in nx.weakly_connected_components(G1):
        nx.draw(c)
    plt.show()



def tab8():
    # mlnode = pandas.read_csv('mlnode.csv')
    # mledge = pandas.read_csv('mledge.csv')
    # scamNodeDegree = pandas.read_csv('scamNodeDegree.csv')
    # scamNodeDegreeList = list(scamNodeDegree['Id'])
    # scamNodeDegreeEdge = pandas.read_csv('scamNodeDegreeEdge.csv')
    # G1 = nx.DiGraph()
    # G1.add_nodes_from(scamNodeDegreeList)
    # edges = []
    # for index,row in scamNodeDegreeEdge.iterrows():
    #     mytuple = (row['Source'],row['Target'],row['Weight'])
    #     edges.append(mytuple)
    # G1.add_weighted_edges_from(edges)
    # num = 0
    # weakly_connected_components12 = []
    # for c in nx.weakly_connected_components(G1):
    #     weakly_connected_components12.append(c)
    #     num += 1
    # with open('weakly_connected_components12.txt','w') as f:
    #     print(weakly_connected_components12,file=f)
    # print(num)
    # plt.subplot(111)
    # nx.draw(G1,node_size=10)
    # plt.show()
    with open('scamInTx.txt','r') as f:
        scamInTx = literal_eval(f.read())
    with open('scamOutTx.txt','r') as f:
        scamOutTx = literal_eval(f.read())
    with open('weakly_connected_components12.txt','r') as f:
        weakly_connected_components12 = literal_eval(f.read())#集合的列表
    index2len = {}
    for index,group in enumerate(weakly_connected_components12):
        index2len[index] = len(group)
    zipped = zip(index2len.keys(),index2len.values())
    index2len = sorted(zipped,key=lambda x:(x[1]))
    print(index2len)

    indexlist = [50,5,41,8,4]
    for index in indexlist:
        maxoutdegree = 0
        maxaddr = ''
        for addr in weakly_connected_components12[index]:
            if addr in scamOutTx.keys() and scamOutTx[addr] > maxoutdegree:
                maxoutdegree = scamOutTx[addr]
                maxaddr = addr
        print(maxaddr)
        print(scamInTx[maxaddr])
        print(scamOutTx[maxaddr])
    #0x83053c32b7819f420dcfed2d218335fe430fe3b5,0x604da69d33a0df6414c30270434ba4e0e95d8cf6,0x3681828da105fc3c44e212f6c3dc51a0a5a6f5c6,
    # 0xb3bf9e759c859a63176feaa2244517b2c9309edb，0xa09871aeadf4994ca12f5c0b6056bbd1d343c029
    # if len(group) < 5:
        #     num += 1
    # print(num)
    #349 / 393 = 89%
    #根据每个弱连通图的地址查找最大outdegree的节点，最多节点的五个组里挑最多出度的节点

def fig13evo():
    #如何通过拓展得到community，和某个欺诈地址进行交易的可疑地址作为同个community
    addrlist = ['0x83053c32b7819f420dcfed2d218335fe430fe3b5',
                '0x0111a93692b5ccff5c4b006d88d82664f76dfd21',
                '0xb1898895a2ac9ef09afeda89ee559427b7142f2e',
                '0x7f183cce61efcabb8d2eb1340f2a94c5eb1bf8e5',
                '0x408fac2d2d3d3ac4e356aa188119f5624fd8c9f6',
                '0x13098fdefba2129b8873bcff5c9ce2de1cf7f32f',
                '0x24ecad110119e2c763f1afc7f4e5dd159238d143',
                '0xb70f65c1faa2f1160d006b585ebf564781e8dae6',
                '0xe7d60131d3773a8cdf75162c2cb8be459600d9b2',
                '0x2b065809f6ec6df32878bcd26711a0e2bcf59c26',
                '0x3da2bb8c828643d13c399e71d9c7b70852d45492',
                '0xa9d499f4e14d5046cbdfd01cde3e8c3ea0ac94d7',
                '0xe8ddafab7212f988d84c973471c6feab851769e8',
                '0x0084515449b037205a33d6d3940a5684126aa4b5',
                '0xb7686c343be859d7875f8d6bb7c228858e229b8d',
                '0x5416400b0d5798dfdcefa2d22335f18982ee22fa',
                '0x1ea3ecc937bae79353d198db2cdab1e5cbe625dd',
                '0xe0430890f9c7010ad6743293fc352680e2f01b4e',
                '0xdafc2a2d4e070f863cfcb5887bc2516e05e18877',
                '0xe7860fd151cbbad28141dfd4cc7cd6d090e4ead8',
                '0x7970780b07b6a8c814eaeab01a09c4f922906da4',
                '0xb3958757fd6d2dfbcc934f39faa1f82210da6f50',
                '0xf9c8c57fe9eaabc5dfef8a8d03ef308f2a28c91c',
                '0x7fa4cbcfed0362516ab52c16d4fde35a72a5c829',
                '0x1ee6b25a0e3f855a10916766df3a056e924e5dce',
                '0xbbdef8b12babd3ab6018566bc17ec3aa302b8348',
                ]
    # 29056
    # 766
    # 76
    # 906
    # 4688
    # 3232
    # 2210
    # 1016
    # 1136
    # 1038
    # 1244
    # 3602
    # 626
    # 960
    # 6884
    # 22
    # 234
    # 684
    # 4516
    # 24
    # 1564
    # 13908
    # 4300
    # 1228
    # 5598
    # 3344
    #一个一个community算，在mledge（不需要在mledge2找）中找和代表的欺诈地址列表有交易的地址，动态添加某个community的欺诈地址列表，比较快的办法用csv的布尔搜索，迭代一次
    mledge = pandas.read_csv('mledge.csv')
    mlnode = pandas.read_csv('mlnode.csv')
    mledge0 = mledge[(mledge['Source'].notnull()) & (mledge['Source']==addrlist[0]) | (mledge['Target'].notnull()) & (mledge['Target']==addrlist[0])]
    list0 = list(mledge0['Source']) + list(mledge0['Target'])
    mledge0 = mledge[(mledge['Source'].notnull()) & (mledge['Source'].isin(list0)) | (mledge['Target'].notnull()) & (mledge['Target'].isin(list0))]
    list0 = list(mledge0['Source']) + list(mledge0['Target'])

def nxfig13():
    len1 = [29056,4688,3602,6884,4516,13908,4300,1228,5598,3344]
    len2 = [816,16,11,10,9,9,9,8,8,8]
    x = numpy.array([1,2,3,4,5,6,7,8,9,10])
    width = 0.2
    plt.bar(x,len1,width,label='After Expansion',color='black')
    plt.bar(x+width,len2,width,label='Before Expasion',color='gray')
    for a, b in zip(x, len1):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=6)
    x = x + width
    for a, b in zip(x, len2):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=6)
    plt.yscale('log')
    plt.legend()
    plt.savefig('fig13.jpg')
    plt.show()
def fig14():#用nx画fig14,但是未知节点需要被欺诈节点转账,改为用gephi画
    # mlnode = pandas.read_csv('mlnode.csv')
    # mledge = pandas.read_csv('mledge.csv')
    # mledge2 = pandas.read_csv('mledge2.csv')
    # mlnode2 = pandas.read_csv('mlnode2.csv')
    # print(len(mlnode))
    # print(len(mledge))
    # mlnode14 = mlnode[(mlnode['type']!='normal')]#欺诈交易中的非正常节点
    # # mlnode142 = mlnode2[(mlnode2['type']!='normal')]#正常交易中的非正常节点
    # # frames = [mlnode14,mlnode142]
    # # mlnode14 = pandas.concat[frames]
    # mlnodelist = list(mlnode14['Id'])#欺诈节点交易列表排除正常节点后的节点，包括欺诈节点和未知节点，要么欺诈要么未知，但是未知节点需要被欺诈节点转账
    # mlnodescam = mlnode14[(mlnode14['type']=='scam')]
    # mlnodescamlist = list(mlnodescam['Id'])#edge导入gephi，如果边的节点不在mlnode14中则会显示null类
    # mledge14 = mledge[(mledge['Source'].notnull()) & (mledge['Source'].isin(mlnodelist)) & (mledge['Target'].notnull()) & (mledge['Target'].isin(mlnodelist))]#两种交易的所有和非正常节点有关的节点
    # # mledge142 = mledge2[(mledge2['Source'].notnull()) & (mledge2['Source'].isin(mlnodelist)) & (mledge2['Target'].notnull()) & (mledge2['Target'].isin(mlnodelist))]
    # # frames = [mledge14,mledge142]
    # # mledge14 = pandas.concat(frames)
    # mlnode14.to_csv('mlnodefig14.csv')#两种交易中的所有非正常节点，需要改为欺诈节点和可疑节点
    # mledge14.to_csv('mledgefig14.csv')#两种交易中起点终点都是非正常节点的交易
    # mledgescam14 = mledge14[(mledge14['Source'].notnull()) & (mledge14['Source'].isin(mlnodescamlist)) | (mledge14['Target'].notnull()) & (mledge14['Target'].isin(mlnodescamlist))]
    # mlnodeSusp = list(set(list(mledgescam14['Source']) + list(mledgescam14['Target'])))#根据相邻有罪原则得到的新的欺诈节点
    # print(len(mlnodeSusp))
    # return

    # mlnode14 = pandas.read_csv('mlnodefig14.csv')#这两个统计没有问题
    # mledge14 = pandas.read_csv('mledgefig14.csv')
    # G1 = nx.DiGraph()
    # ScamNodeList = []
    # ScamEdgeList = []
    # UnknownNodeList = []
    # UnknownEdgeList = []
    # colorMap = []
    # specialNodeList = list(mlnode14['Id'])
    # for index,row in mlnode14.iterrows():
    #     if row['type'] == 'scam':
    #         ScamNodeList.append(row['Id'])
    #         colorMap.append('yellow')#后来没有用colormap画图，但是用到了UnknownNodeList，这个统计是有问题的
    #     if row['type'] == 'unknown':
    #         UnknownNodeList.append(row['Id'])
    #         colorMap.append('pink')
    # edges = []
    # for index,row in mledge14.iterrows():
    #     mytuple = (row['Source'],row['Target'],row['Weight'])
    #     edges.append(mytuple)
    #     if row['Source'] in ScamNodeList or row['Target'] in ScamNodeList:
    #         ScamEdgeList.append((row['Source'],row['Target'],))
    #     if row['Source'] in UnknownNodeList or row['Target'] in UnknownNodeList:
    #         UnknownEdgeList.append((row['Source'],row['Target'],))
    #
    # with open('mlnodefig14SpecialNode.txt','w') as f:
    #     print(specialNodeList,file=f)
    # with open('mlnodefig14Edges.txt','w') as f:
    #     print(edges,file=f)#简化后的欺诈交易和正常交易的边
    # with open('mlnodefig14ScamNode.txt','w') as f:
    #     print(ScamNodeList,file=f)
    # with open('mlnodefig14UnknownNode.txt','w') as f:
    #     print(UnknownNodeList,file=f)
    # with open('mlnodefig14ScamEdge.txt','w') as f:
    #     print(ScamEdgeList,file=f)
    # with open('mlnodefig14UnknownEdge.txt','w') as f:
    #     print(UnknownEdgeList,file=f)
    # return


    mlnode14 = pandas.read_csv('mlnodefig14.csv')
    mledge14 = pandas.read_csv('mledgefig14.csv')

    with open(r'C:\Users\ljh\Desktop\fig14node.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'type', 'indegree', 'outdegree', 'degree'])
        for index, row in mlnode14.iterrows():#打印的是欺诈交易和正常交易的
            writer.writerow([row['Id'], row['type'], row['indegree'], row['outdegree'], row['degree']])
    with open(r'C:\Users\ljh\Desktop\fig14edge.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Source', 'Target', 'Weight'])
        for index, row in mledge14.iterrows():
            writer.writerow([row['Source'], row['Target'], row['Weight']])
    return

    with open('mlnodefig14SpecialNode.txt','r') as f:
        specialNodeList = literal_eval(f.read())
    with open('mlnodefig14Edges.txt','r') as f:
        edges = literal_eval(f.read())
    with open('mlnodefig14ScamNode.txt','r') as f:
        ScamNodeList = literal_eval(f.read())
    with open('mlnodefig14UnknownNode.txt','r') as f:
        UnknownNodeList = literal_eval(f.read())
    with open('mlnodefig14ScamEdge.txt','r') as f:
        ScamEdgeList = literal_eval(f.read())
    with open('mlnodefig14UnknownEdge.txt','r') as f:
        UnknownEdgeList = literal_eval(f.read())

    G1 = nx.DiGraph()
    G1.add_nodes_from(specialNodeList)
    G1.add_weighted_edges_from(edges)

    # pos = nx.fruchterman_reingold_layout(G1,k=0.5)
    pos = nx.spring_layout(G1)#kind=2

    plt.subplot(111)
    nx.draw_networkx_nodes(G1,pos,nodelist=ScamNodeList,node_color='yellow',node_size=1,label='Scam')
    nx.draw_networkx_nodes(G1,pos,nodelist=UnknownNodeList,node_color='pink',node_size=1,label='Unknown')
    nx.draw_networkx_edges(G1,pos,edgelist=ScamEdgeList,width=0.1,edge_color='yellow',arrowsize=1,arrowstyle='-',arrows=False)
    nx.draw_networkx_edges(G1,pos,edgelist=UnknownEdgeList,width=0.1,edge_color='pink',arrowsize=1,arrowstyle='-',arrows=False)
    plt.legend()
    plt.savefig('fig14.jpg')
    plt.show()

    #166513个节点
    #197687边
    #132129个可疑节点
def rightfig14():#统计好可疑节点后用gephi画图而不是nx
    mlnode14 = pandas.read_csv('mlnodefig14.csv')#非正常节点和节点种类（欺诈和未知），根据正常交易和欺诈交易统计的
    mledge14 = pandas.read_csv('mledgefig14.csv')#
    ScamNodeList = []
    ScamEdgeList = []
    UnknownNodeList = []#未知的可疑节点
    UnknownSusEdgeList = []#和未知可疑节点有关的交易
    mlnode14list = list(mlnode14['Id'])#非正常节点，需要先统计由欺诈节点转账的‘可疑节点’，根据addr.txt和mledge.csv和mledge2.csv也就是mledge14.csv统计
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('mlnodefig14ScamNode.txt','r') as f:
        ScamNodeList = literal_eval(f.read())
    with open('mlnodefig14UnknownNode.txt','r') as f:
        UnknownNodeList = literal_eval(f.read())
    # for index,row in mlnode14.iterrows():
    #     if row['type'] == 'scam':
    #         ScamNodeList.append(row['Id'])
    #     if row['type'] == 'unknown':
    #         UnknownNodeList.append(row['Id'])#统计正常交易和欺诈交易中的未知节点，下一行找未知的可疑节点，可以直接使用统计好的结果
    UnknownSusRel = mledge14[(mledge14['Source'].notnull()) & (mledge14['Source'].isin(addrlist)) & (mledge14['Target'].notnull()) & (mledge14['Target'].isin(UnknownNodeList))]  # 转账接收节点是未知的
    UnknownSusNodeList = list(set(list(UnknownSusRel['Source']) + list(UnknownSusRel['Target'])))#根据找到的可疑交易统计可疑节点
    # for index,row in mledge14.iterrows():#开销最大的操作
    #     if row['Source'] in ScamNodeList or row['Target'] in ScamNodeList:
    #         ScamEdgeList.append((row['Source'],row['Target'],))
    #     if row['Source'] in UnknownSusNodeList or row['Target'] in UnknownSusNodeList:
    #         UnknownSusEdgeList.append((row['Source'],row['Target'],))
    # # with open('mlnodefig14ScamNode.txt','w') as f:
    # #     print(ScamNodeList,file=f)
    # with open('mlnodefig14UnknownSusNode.txt','w') as f:#未知可疑节点
    #     print(UnknownSusNodeList,file=f)
    # # with open('mlnodefig14ScamEdge.txt','w') as f:#欺诈节点和交易统计没出问题不用重新写
    # #     print(ScamEdgeList,file=f)
    # with open('mlnodefig14UnknownSusEdge.txt','w') as f:
    #     print(UnknownSusEdgeList,file=f)
    with open('mlnodefig14UnknownSusNode.txt','r') as f:
        UnknownSusNodeList = literal_eval(f.read())
    with open('mlnodefig14UnknownSusEdge.txt','r') as f:
        UnknownSusEdgeList = literal_eval(f.read())
    # print(len(UnknownSusNodeList))#33664
    # print(len(UnknownSusEdgeList))#111203
    scamAndSusNodeList = addrlist + UnknownSusNodeList
    scamAndSusNodeCsv = mlnode14[mlnode14['Id'].isin(scamAndSusNodeList)]#欺诈和可疑节点列表
    scamAndSusNodeCsv.to_csv('scamAndSusNode.csv')
    scamAndSusEdge = mledge14[(mledge14['Source'].notnull()) & (mledge14['Source'].isin(scamAndSusNodeList)) & (mledge14['Target'].notnull()) & (mledge14['Target'].isin(scamAndSusNodeList))]
    scamAndSusEdge.to_csv('scamAndSusEdge.csv')
    #先统计欺诈节点和可疑节点
    with open(r'C:\Users\ljh\Desktop\fig14susNode.csv', 'w', newline='') as f:#写入桌面的gephi的csv文件画图
        writer = csv.writer(f)
        writer.writerow(['Id', 'type', 'indegree', 'outdegree', 'degree'])#35319
        # scamAndSusNodeCsv = pd.read_csv('scamAndSusNode.csv')
        for index, row in scamAndSusNodeCsv.iterrows():
            writer.writerow([row['Id'], row['type'], row['indegree'], row['outdegree'], row['degree']])
    with open(r'C:\Users\ljh\Desktop\fig14susEdge.csv', 'w', newline='') as f:#写入桌面的gephi的csv文件画图
        writer = csv.writer(f)
        writer.writerow(['Source', 'Target', 'Weight'])
        # scamAndSusEdge = pd.read_csv('scamAndSusEdge.csv')
        for index, row in scamAndSusEdge.iterrows():
            writer.writerow([row['Source'], row['Target'], row['Weight']])#38935 45084

def fig15():
    scamNodeDegree = pandas.read_csv('scamNodeDegree.csv')
    mledge = pandas.read_csv('mledge.csv')
    mlnode15 = pandas.read_csv('mlnode.csv')
    degreeScamnode = list(scamNodeDegree['Id'])#根据原来的community进行筛选
    mledge15 = mledge[(mledge['Source'].notnull()) & (mledge['Source'].isin(degreeScamnode)) | (mledge['Target'].notnull()) & (mledge['Target'].isin(degreeScamnode))]
    mledgeSource = list(mledge['Source'])
    mledgeTarget = list(mledge['Target'])
    mledgelist = mledgeSource + mledgeTarget
    # mlnode15 = mlnode[(mlnode['Id'].isin(mledgelist))]#拓展后和原来有区别
    mlnode15.to_csv('scamNodeDegree15.csv')
    mledge15.to_csv('scamNodeDegreeEdge15.csv')
    #92768
def fig15x():#community拓展后的size
    # with open('weakly_connected_components12.txt','r') as f:
    #     weakly_connected_components12 = literal_eval(f.read())
    # mledge = pandas.read_csv('mledge.csv')
    # for index,weakset in enumerate(weakly_connected_components12):
    #     expansionEdge = mledge[(mledge['Source'].notnull()) & (mledge['Source'].isin(weakset)) | (mledge['Target'].notnull()) & (mledge['Target'].isin(weakset))]
    #     expansionNode = set(list(expansionEdge['Source']) + list(expansionEdge['Target']))
    #     weakly_connected_components12[index] = weakset.union(expansionNode)
        # print(index)
    #     print(weakly_connected_components12[index])
    # print(weakly_connected_components12)
    # with open('weakly_connected_components15.txt','w') as f:
    #     print(weakly_connected_components12,file=f)
    with open('weakly_connected_components15.txt','r') as f:
        weakly_connected_components15 = literal_eval(f.read())
    num = 0
    for weakset in weakly_connected_components15:
        if len(weakset) < 5:
            num += 1
    print(num)
    # 14 / 393 = 0.03
def fig15a():
    # mledge = pandas.read_csv('mledge.csv')
    # mlnode = pandas.read_csv('mlnode.csv')
    # addrlist = list(mledge['Source']) + list(mledge['Target'])#欺诈地址相连的所有地址，如果需要精确到unknown节点可以再从mlnode中筛选
    # 所有涉及的地址进行计算，然后再根据group放到每个地址的字典中
    # df1 = pandas.read_csv('ntx.csv')
    # df2 = pandas.read_csv('itx.csv')
    # frames = [df1, df2]
    # df = pandas.concat(frames)
    # #不能转换为月份，要计算living hours
    # df = df.sort_values('timeStamp')

    # fig15addr2timestamp = {}
    # for addr in addrlist:
    #     fig15addr2timestamp[addr] = []
    # for index, row in df.iterrows():
    #     if isinstance(row['to'], str) and row['to'] in addrlist:
    #         fig15addr2timestamp[row['to']].append(row['timeStamp'])#没有对时间戳排序又错了
    # with open('fig15addr2timestamp.txt', 'w', encoding='utf-8') as f:#从地址的living time到欺诈group的living time
    #     print(fig15addr2timestamp,file=f)

    # fig15addr2livinghours = {}#小时数
    # for addr, time in fig15addr2timestamp.items():
    #     if len(time) > 0:
    #         start = datetime.datetime.fromtimestamp(time[0])
    #         end = datetime.datetime.fromtimestamp(time[-1])
    #         fig15addr2livinghours[addr] = (end - start).days * 24 + (end - start).seconds / 3600
    # with open('fig15addr2livinghours.txt','w') as f:
    #     print(fig15addr2livinghours,file=f)

    # with open('fig15addr2timestamp.txt', 'r', encoding='utf-8') as f:
    #     fig15addr2timestamp = literal_eval(f.read())
    with open('fig15addr2livinghours.txt', 'r', encoding='utf-8') as f:
        fig15addr2livinghours = literal_eval(f.read())
    addr2day = {}
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist2 = literal_eval(f.read())
    for addr,living in fig15addr2livinghours.items():
        if addr not in addrlist2:
            addr2day[addr] = living / 24
    # print(addr2day)
    # addr2dayvalues = list(addr2day.values())
    # print(max)
    zipped = zip(addr2day.keys(), addr2day.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[1]))
    print(sort_zipped)
    living2num = {}
    for addr,living in addr2day.items():
        living2num[living] = living2num.get(living,0) + 1
    print(len(fig15addr2livinghours))
    zipped = zip(living2num.keys(), living2num.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    # print(sort_zipped)
    result = zip(*sort_zipped)
    plt.figure(figsize=(6, 6.5))
    x, y = [list(x) for x in result]
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x, percentage)
    x_major_locator = MultipleLocator(200)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.set_ylabel('CDF of Addresses')
    ax.set_xlabel('the living time of address(Days)')
    ax.set_xlim(1)
    plt.savefig('fig15a.jpg')
    plt.show()

def fig15b():
    # mledge = pandas.read_csv('mledge.csv')
    # mlnode = pandas.read_csv('mlnode.csv')
    # addrlist = list(mledge['Source']) + list(mledge['Target'])#欺诈地址相连的所有地址，如果需要精确到unknown节点可以再从mlnode中筛选
    # df1 = pandas.read_csv('ntx.csv')
    # df2 = pandas.read_csv('itx.csv')
    # frames = [df1, df2]
    # df = pandas.concat(frames)
    # df['timeStamp'] = df['timeStamp'].map(lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y-%m"))
    # df = df.sort_values('timeStamp')
    # addr2time = {}
    # for addr in addrlist:
    #     addr2time[addr] = []
    # for index, row in df.iterrows():
    #     if isinstance(row['to'], str) and row['to'] in addrlist:
    #         addr2time[row['to']].append(row['timeStamp'])
    # with open('fig15addr2time.txt', 'w', encoding='utf-8') as f:
    #     print(addr2time,file=f)
    with open('fig15addr2time.txt', 'r', encoding='utf-8') as f:
        addr2time = literal_eval(f.read())
    with open('fig15addr2mon.txt', 'r', encoding='utf-8') as f:#还可以用于gephi的做法
        addr2mon = literal_eval(f.read())
    # for addr, time in addr2time.items():
    #     time.sort()
    # addr2mon = {}
    # for addr, time in addr2time.items():
    #     if len(time) > 0:
    #         addr2mon[addr] = time[0]
    # with open('fig15addr2mon.txt', 'w', encoding='utf-8') as f:
    #     print(addr2mon,file=f)
    #每个group查询appearance time，确定每个group的最早时间和living time（计算方法改变），先求最早时间取最值
    with open('weakly_connected_components15.txt','r') as f:#根据拓展一次后的scam group分组进行计算
        weakly_connected_components15 = literal_eval(f.read())
    group2appearance = {}#每个group的首次出现时间
    for index,weakset in enumerate(weakly_connected_components15):
        group2appearance[index] = []
        for addr in weakset:
            if addr in addr2mon.keys():#通过to地址才计算timestamp会遗漏一部分只是输出的地址
                group2appearance[index].append(addr2mon[addr])
        group2appearance[index].sort()
    # print(group2appearance)#不同组的每个地址的最先出现时间
    group2single = {}
    for index,appearlist in group2appearance.items():
        group2single[index] = appearlist[0]
    # print(group2single)
    mon2count = {}
    for index,mon in group2single.items():
        mon2count[mon] = mon2count.get(mon,0) + 1
    x = mon2count.keys()
    y = mon2count.values()
    zipped = zip(x, y)
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    plt.figure(figsize=(6, 6.5))
    fig, ax = plt.subplots()
    ax.set_ylabel('Number Of Scam Group')
    ax.set_xlabel('Appearance Date')
    plt.xticks(rotation=30)
    x_major_locator = MultipleLocator(3)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.plot(x,y)
    plt.savefig('fig15b.jpg',bbox_inches='tight',pad_inches=0.2)
    plt.show()
    # with open('fig15b.txt','w') as f:
    #     print(mon2count,file=f)
def nxfig16a():#不是正常地址的收入，是欺诈地址的收入
    # df1 = pandas.read_csv('ntx.csv')
    # df2 = pandas.read_csv('itx.csv')
    # frames = [df1, df2]
    # df = pandas.concat(frames)
    with open('weakly_connected_components15.txt','r') as f:#根据拓展一次后的scam group分组进行计算
        weakly_connected_components15 = literal_eval(f.read())
    # # eth = 2460.268204521556
    # addr2profit = {}
    # mledge = pandas.read_csv('mledge.csv')
    # addrlist = list(mledge['Source']) + list(mledge['Target'])#得出所有需要计算利润的地址，然后计算每个地址的利润，再分给每个组
    # #将scamgroup扁平化得到所有的地址当作一个需要排除的列表，因为不同的group之间不会互相转账，同一个group的转账不能用于计算利润
    # # #每个group逐条地址进行计算，不能这样算，因为同一个group的可能重复计算
    # scamgroupAddr = []
    # for group in weakly_connected_components15:
    #     scamgroupAddr += list(group)
    # for addr in scamgroupAddr:
    #     addr2profit[addr] = 0
    # for index, row in df.iterrows():
    #     if isinstance(row['to'],str) and row['to'].lower() in scamgroupAddr and isinstance(row['from'],str) and row['from'].lower() not in scamgroupAddr:
    #         try:
    #             if isinstance(row['value'],str):
    #                 addr2profit[row['to']] += int(row['value']) / 1000000000000000000
    #         except Exception:
    #             print(row['value'])
    #             print(row['to'])
    #             traceback.print_exc()
    # with open('fig16addr2profit2.txt', 'w') as f:#得到抓取的正常地址和异常地址的以太币利润
    #     print(addr2profit,file=f)
    # return
    group2profit = {}
    # profit2num = {}
    with open('fig16addr2profit2.txt', 'r') as f:
        addr2profit = literal_eval(f.read())
    # with open('unknownAddrProfit.txt','r') as f:
    #     addr2profit2 = literal_eval(f.read())
    # for addr,profit in addr2profit.items():
    #     addr2profit[addr] = round(addr2profit[addr] / eth)
    # for addr,profit in addr2profit.items():
    #     profit2num[profit] = profit2num.get(profit,0) + 1
    # print(profit2num)
    # print(len(profit2num))
    #计算利润和数量的映射
    for index,weakset in enumerate(weakly_connected_components15):
        group2profit[index] = 0
        for addr in weakset:
            group2profit[index] += addr2profit[addr]
    # with open('fig16group2profit.txt', 'w') as f:
    #     print(group2profit, file=f)
    profit2num = {}
    for group,profit in group2profit.items():
        profit2num[profit] = profit2num.get(profit,0) + 1
    # print(profit2num)
    # print(len(profit2num))
    x = profit2num.keys()
    y = profit2num.values()
    zipped = zip(x, y)
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    plt.figure(figsize=(6, 6.5))
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x, percentage)
    ax.set_ylabel('CDF of Addresses')
    ax.set_xlabel('Profit(Eth)')
    # x_major_locator = MultipleLocator(5000)
    # ax.xaxis.set_major_locator(x_major_locator)
    # ax.set_xlim(1)
    plt.xscale('log')
    plt.savefig('fig16a.jpg')
    plt.show()

def nxfig16b():
    # df1 = pandas.read_csv('ntx.csv')
    # df2 = pandas.read_csv('itx.csv')
    # frames = [df1, df2]
    # df = pandas.concat(frames)
    # mledge = pandas.read_csv('mledge.csv')
    # mlnode = pandas.read_csv('mlnode.csv')
    # addrlist = list(mledge['Source']) + list(mledge['Target'])
    with open('weakly_connected_components15.txt','r') as f:#计算每个group的from地址数，排除同一个group互相转账的交易
        weakly_connected_components15 = literal_eval(f.read())
    # scamgroupAddr = []
    # addr2vitctim = {}
    # for group in weakly_connected_components15:
    #     scamgroupAddr += list(group)
    # for addr in scamgroupAddr:
    #     addr2vitctim[addr] = []
    # for index, row in df.iterrows():
    #     if isinstance(row['to'],str) and row['to'].lower() in scamgroupAddr and isinstance(row['from'],str) and row['from'].lower() not in scamgroupAddr:
    #         try:
    #             addr2vitctim[row['to']].append(row['from'])
    #         except Exception:
    #             print(row['to'])
    #             traceback.print_exc()
    # with open('fig16addr2victim.txt', 'w') as f:
    #     print(addr2vitctim,file=f)
    # return

    # with open('fig16addr2victim.txt', 'r') as f:
    #     addr2vitctim = literal_eval(f.read())
    # group2victim = {}
    # for index,weakset in enumerate(weakly_connected_components15):
    #     group2victim[index] = []
    #     for addr in weakset:
    #         if addr in addr2vitctim.keys():
    #             group2victim[index] += addr2vitctim[addr]
    #     group2victim[index] = list(set(group2victim[index]))
    # with open('fig16group2victim.txt', 'w') as f:
    #     print(group2victim,file=f)
    # return
    with open('fig16group2victim.txt', 'r') as f:
        group2victim = literal_eval(f.read())
    for index,victim in group2victim.items():
        group2victim[index] = len(victim)
    print(group2victim)
    victim2num = {}
    for index,victim in group2victim.items():
        victim2num[victim] = victim2num.get(victim,0) + 1
    print(victim2num)
    x = victim2num.keys()
    y = victim2num.values()
    zipped = zip(x, y)
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    print(sort_zipped)
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    plt.figure(figsize=(6, 6.5))
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x, percentage)
    ax.set_ylabel('CDF of Addresses')
    ax.set_xlabel('# of victim')
    # x_major_locator = MultipleLocator(1000)
    # ax.xaxis.set_major_locator(x_major_locator)
    ax.set_xlim(1)
    plt.savefig('fig16b.jpg')
    plt.show()
def xlsx2csv():
    dataxls = pd.read_excel('fig12.xlsx')
    dataxls.to_csv('fig12.csv')
def gephifig12():
    # df = pd.read_csv('fig12.csv')
    # groupdict = {}
    # groupnum = 397
    # for i in range(0,groupnum):
    #     groupdict[i] = []
    # for index,row in df.iterrows():
    #     group = row['class']
    #     groupdict[group].append(row['Id'])
    # with open('gephiGroup.txt','w') as f:
    #     print(groupdict,file=f)
    with open('gephiGroup.txt','r') as f:
        groupdict = literal_eval(f.read())
def gephifig15a():#根据gephi的分组统计living time和用nx统计的living time一样
    df1 = pandas.read_csv('ntx.csv')
    df2 = pandas.read_csv('itx.csv')
    frames = [df1, df2]
    df = pandas.concat(frames)
    #不能转换为月份，要计算living hours
    df = df.sort_values('timeStamp')
    # mledge = pandas.read_csv('mledge.csv')
    # mlnode = pandas.read_csv('mlnode.csv')
    # addrlist = list(mledge['Source']) + list(mledge['Target'])#欺诈地址相连的所有地址，如果需要精确到unknown节点可以再从mlnode中筛选
    # fig15addr2timestamp = {}
    # for addr in addrlist:
    #     fig15addr2timestamp[addr] = []
    # for index, row in df.iterrows():
    #     if isinstance(row['to'], str) and row['to'] in addrlist:#
    #         fig15addr2timestamp[row['to']].append(row['timeStamp'])#已经排序了时间戳
    # with open('fig15addr2timestamp.txt', 'w', encoding='utf-8') as f:#从地址的living time到欺诈group的living time
    #     print(fig15addr2timestamp,file=f)
    #
    # fig15addr2livinghours = {}
    # for addr, time in fig15addr2timestamp.items():
    #     if len(time) > 0:
    #         start = datetime.datetime.fromtimestamp(time[0])
    #         end = datetime.datetime.fromtimestamp(time[-1])
    #         fig15addr2livinghours[addr] = (end - start).days * 24 + (end - start).seconds / 3600#计算欺诈交易涉及的每个地址的living hours
    # #上面算的是欺诈节点或者正常节点的living time，这里还有未知节点的需要计算
    # with open('fig15addr2livinghours.txt','w') as f:
    #     print(fig15addr2livinghours,file=f)

    with open('fig15addr2timestamp.txt', 'r', encoding='utf-8') as f:
        fig15addr2timestamp = literal_eval(f.read())
    with open('fig15addr2livinghours.txt', 'r', encoding='utf-8') as f:#原来用nx的分组统计的living time还可以用
        fig15addr2livinghours = literal_eval(f.read())
    addr2day = {}
    for addr, living in fig15addr2livinghours.items():
        addr2day[addr] = living / 24
    living2num = {}
    for addr, living in addr2day.items():
        living2num[living] = living2num.get(living, 0) + 1
    print(len(fig15addr2livinghours))
    zipped = zip(living2num.keys(), living2num.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    plt.figure(figsize=(6, 6.5))
    x, y = [list(x) for x in result]
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x, percentage)
    x_major_locator = MultipleLocator(200)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.set_ylabel('CDF of Addresses')
    ax.set_xlabel('the living time of address')
    ax.set_xlim(1)
    plt.savefig('fig15a.jpg')
    plt.show()

def gephifig16a():
    with open('fig16addr2profit2.txt', 'r') as f:
        addr2profit = literal_eval(f.read())
    with open('unknownAddrProfit.txt', 'r') as f:
        unknownAddrProfit = literal_eval(f.read())
    # with open('gephiGroup.txt','r') as f:
    #     groupdict = literal_eval(f.read())
    with open('fig12exp2FromTo.txt','r') as f:
        groupdict = literal_eval(f.read())
    group2profit = {}
    for index,group in groupdict.items():
        group2profit[index] = 0
    for index,group in groupdict.items():
        # group2profit[index] = 0
        for addr in group:
            if addr in unknownAddrProfit.keys():#计算拓展后的组的利润
                group2profit[index] += unknownAddrProfit[addr]
            elif addr in addr2profit.keys():
                group2profit[index] += addr2profit[addr]
            # print(addr2profit[addr])
    with open('gephiGroup2profit.txt','w') as f:
        print(group2profit,file=f)
    profit2num = {}
    for group, profit in group2profit.items():
        profit2num[profit] = profit2num.get(profit, 0) + 1
    # print(profit2num)
    # return
    x = profit2num.keys()
    y = profit2num.values()
    zipped = zip(x, y)
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    plt.figure(figsize=(6, 6.5))
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x, percentage)
    ax.set_ylabel('CDF of Addresses')
    ax.set_xlabel(' The profit distribution of scam group')
    # x_major_locator = MultipleLocator(5000)
    # ax.xaxis.set_major_locator(x_major_locator)
    # ax.set_xlim(1)
    plt.xscale('log')
    plt.savefig('gephifig16a.jpg')
    plt.show()

def fig16aStats():
    with open('gephiGroup2profit.txt','r') as f:
        gephiGroup2profit = literal_eval(f.read())
    num = 0
    for index,profit in gephiGroup2profit.items():
        if profit > 100:
            num += 1
    print(num)
    print(num/397)
    zipped1 = zip(gephiGroup2profit.keys(), gephiGroup2profit.values())
    gephiGroup2profit = sorted(zipped1, key=lambda x: (x[1]))
    print(gephiGroup2profit)
    # print(gephiGroup2profit.index(max(gephiGroup2profit.values())))
    #计算第192组的地址数
    with open('fig12exp2FromTo.txt','r') as f:#欺诈地址相关交易的每个地址的交易对象
        exp = literal_eval(f.read())#每个组拓展后的地址
    num = len(exp[192])
    profit = gephiGroup2profit[192][1]
    print(profit/num)
def scamUnknownVictim():
    df1 = pandas.read_csv('ntxFromTo.csv')
    df2 = pandas.read_csv('itxFromTo.csv')
    # with open('unknownAddrVictim.txt','r') as f:
    #     unknownAddrVictim = literal_eval(f.read())
    scamaddr2victim = defaultdict(list)
    scamaddr2addr = defaultdict(list)
    scamVictimNum = defaultdict(int)
    unknownAddrVictimNum = defaultdict(int)
    for index, row in df1.iterrows():
        if isinstance(row['from'], str) and isinstance(row['to'], str):
            scamaddr2victim[row['to']].append(row['from'])
    for index, row in df2.iterrows():
        if isinstance(row['from'], str) and isinstance(row['to'], str):
            scamaddr2victim[row['to']].append(row['from'])
    # with open('scamaddr2addr.csv','w',newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['from','to','value'])#转为用csv统计欺诈地址的交易对象
    #     for index, row in df1.iterrows():
    #         if isinstance(row['from'], str) and isinstance(row['to'], str):
    #             writer.writerow([row['from'],row['to']])
    #     for index, row in df2.iterrows():
    #         if isinstance(row['from'], str) and isinstance(row['to'], str):
    #             writer.writerow([row['from'], row['to']])
    scamaddr2victim = dict(scamaddr2victim)
    scamaddr2addr = dict(scamaddr2addr)
    unknownAddrVictimNum = dict(unknownAddrVictimNum)
    scamVictimNum = dict(scamVictimNum)
    for addr,mylist in scamaddr2victim.items():
        scamVictimNum[addr] = len(scamaddr2victim[addr])
    # for addr,mylist in unknownAddrVictim.items():
    #     unknownAddrVictimNum[addr] = len(unknownAddrVictim[addr])
    with open('scamvictim.txt','w') as f:
        print(scamaddr2victim,file=f)

    with open('scamVictimNum.txt','w') as f:
        print(scamVictimNum,file=f)
    # with open('unknownAddrVictimNum.txt','w') as f:
    #     print(unknownAddrVictimNum,file=f)
# def groupvictim():
#     with open('scamVictimNum.txt','r') as f:
#         scamVictimNum = literal_eval(f.read())
#     with open('unknownAddrVictimNum.txt','r') as f:
#         unknownAddrVictimNum = literal_eval(f.read())
#     with open('fig12exp2FromTo.txt','r') as f:#欺诈地址相关交易的每个地址的交易对象
#         exp = literal_eval(f.read())#每个组拓展后的地址
#     groupvictim = {}
#     for index,addrlist in exp.items():

def beforeGroupProfit():#在拓展之前的组的利润
    with open('gephiGroup.txt','r') as f:
        groupdict = literal_eval(f.read())
    with open('fig16addr2profit2.txt', 'r') as f:
        addr2profit = literal_eval(f.read())
    # group2profit = defaultdict(int)
    group2profit = {}
    for index,group in groupdict.items():
        group2profit[index] = 0
    for index,group in groupdict.items():
        for addr in group:
            group2profit[index] += addr2profit[addr]
            if addr2profit[addr] > 0:
                print(addr2profit[addr])
    mylist = list(group2profit.values())
    print(max(mylist))
    with open('beforeGroupProfit.txt','w') as f:
        print(group2profit,file=f)
def gephifig16b():
    with open('scamVictimNum.txt', 'r') as f:
        scamVictimNum = literal_eval(f.read())
    with open('unknownAddrVictimNum.txt', 'r') as f:
        unknownAddrVictimNum = literal_eval(f.read())
    with open('fig12exp2FromTo.txt','r') as f:#拓展后的组
        fig12exp2FromTo = literal_eval(f.read())
    group2victim = {}
    for index,group in fig12exp2FromTo.items():
        group2victim[index] = 0
    for index,group in fig12exp2FromTo.items():
        # group2profit[index] = 0
        for addr in group:
            if addr in scamVictimNum.keys():
                group2victim[index] += scamVictimNum[addr]
            elif addr in unknownAddrVictimNum.keys():
                group2victim[index] += unknownAddrVictimNum[addr]
            # group2victim[index] += group2victim[addr]
            # print(group2victim[addr])
    # print(group2profit)
    # with open('fig16group2victim.txt','w') as f:
    #     print(group2victim,file=f)
    with open('fig16group2victim.txt','r') as f:
        fig16group2victim = literal_eval(f.read())
    mysum = sum(fig16group2victim.values())
    print(mysum)
    # return
    maxvictim = max(fig16group2victim.values())
    maxindex = list(fig16group2victim.values()).index(maxvictim)
    print(maxvictim)
    print(maxindex)
    victim2num = {}
    for group, victim in group2victim.items():
        victim2num[victim] = victim2num.get(victim, 0) + 1
    # print(profit2num)
    # return
    x = victim2num.keys()#受害者数量
    y = victim2num.values()#
    zipped = zip(x, y)
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    plt.figure(figsize=(6, 6.5))
    fig, ax = plt.subplots()
    cum = numpy.cumsum(y)
    percentage = cum / list(cum)[-1]
    line = ax.plot(x, percentage)
    ax.set_ylabel('CDF of Addresses')
    ax.set_xlabel(' The victim distribution of scam group')
    # x_major_locator = MultipleLocator(5000)
    # ax.xaxis.set_major_locator(x_major_locator)
    # ax.set_xlim(1)
    plt.xscale('log')
    plt.savefig('gephifig16b.jpg')
    plt.show()
def myimg2pdf():
    filedir = os.getcwd()
    filelist = os.listdir(filedir)
    # filename = 'fig6.jpg'
    # fileprefix = filename[:-4]
    # savedFile = filedir + r'\pdf' + '\\' + fileprefix + r'.pdf'
    # with open(savedFile, 'wb') as f:
    #     writeContent = img2pdf.convert(filename)
    #     f.write(writeContent)
    # print(savedFile)
    # return
    for filename in filelist:
        if filename.startswith('fig43') and filename.endswith('.jpg'):
            fileprefix = filename[:-4]
            savedFile = filedir + r'\pdf' + '\\' + fileprefix + r'.pdf'
            with open(savedFile,'wb') as f:
                writeContent = img2pdf.convert(filename)
                f.write(writeContent)
            print(filename)
            print(savedFile)
        # if filename.endswith('.png'):
        #     print(filename)
        #     fileprefix = filename[:-4]
        #     savedFile = filedir + r'\pdf' + '\\' + fileprefix + r'.pdf'
        #     with open(savedFile,'wb') as f:
        #         writeContent = img2pdf.convert(filename)
        #         f.write(writeContent)
        #     print(savedFile)
def getnpz():
    features = np.load('D:\\ether\\testf.npz',allow_pickle=True)
    labels = np.load('D:\\ether\\testlabels.npy',allow_pickle=True)
    adj = np.load('D:\\ether\\testa.npz',allow_pickle=True)
    test = np.load('D:\\ether\\test.npz',allow_pickle=True)
    modified = np.load('D:\\ether\\testmEmbed.npz',allow_pickle=True)
    rgcn = np.load('D:\\ether\\tmp\\test_meta_adj_0.05.npz',allow_pickle=True)
    test = np.load('D:\\ether\\test.npz',allow_pickle=True)
    print(type(modified))
    print(adj['shape'])#原来的17339
    print(features['shape'])#两个连通子图16295
    print(test['adj_shape'])#一个连通子图16289
    print(modified['shape'])
    print(rgcn.files)
    print(len(labels))
    print(features.files)#['indices', 'indptr', 'format', 'shape', 'data']
    print(adj.files)#['indices', 'indptr', 'format', 'shape', 'data']
    print(test.files)#['adj_data', 'adj_indices', 'adj_indptr', 'adj_shape', 'attr_data', 'attr_indices', 'attr_indptr', 'attr_shape', 'class_names', 'labels']
    print(modified.files)#['indices', 'indptr', 'format', 'shape', 'data']
    return
    # np.set_printoptions(threshold=np.inf)
    for entry in features.files:
        print(entry)
        print(features[entry])
        # print(len(features[entry]))
        print('\n')
        # file = 'testmy' + str(entry) + '.txt'
        # with open(file,'w') as f:
        #     print(features[entry],file=f)
    #13752个节点，767个属性，9种分类，长为N的节点标签y的向量，N*D的属性矩阵，N*N的邻接矩阵，N个节点，每个节点有D个属性,245778条边,13381个节点
    #是否需要收集unknown节点的特征
    #这么多节点构建邻接矩阵
    #Amazon，coauthor数据集是被视为无向图

def load_npz(file_name, is_sparse=True):
    with np.load(file_name) as loader:
        # loader = dict(loader)
        if is_sparse:
            adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                 loader['adj_indptr']), shape=loader['adj_shape'])
            if 'attr_data' in loader:
                features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                          loader['attr_indptr']), shape=loader['attr_shape'])
            else:
                features = None
            labels = loader.get('labels')
        else:
            adj = loader['adj_data']
            if 'attr_data' in loader:
                features = loader['attr_data']
            else:
                features = None
            labels = loader.get('labels')
    if features is None:
        features = np.eye(adj.shape[0])
    features = sp.csr_matrix(features, dtype=np.float32)
    return adj, features, labels

def getComponentNode():
    data_filename = 'D:\\ether\\test.npz'
    adj, features, labels = load_npz(data_filename)#用仓库的源代码提取
    # adj = np.load('D:\\ether\\testa.npz',allow_pickle=True)
    # features = np.load('D:\\ether\\testf.npz',allow_pickle=True)
    n_components = 2
    _, component_indices = sp.csgraph.connected_components(adj)  # 使用sp的接口找稀疏矩阵的连通子图，获取子图索引，将有向图转换为无向图找弱连通子图
    component_sizes = np.bincount(component_indices)  # 计算连通子图个数
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending 倒序
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]  # 选择最大连通子图的节点
    with open('componentNode.txt','w') as f:
        print(nodes_to_keep,file=f)
    # with open('test.txt', 'r') as f:
    #     mylist = literal_eval(f.read())
    # for index,alist in enumerate(mylist):
    #     if len(alist) == 0:
    #         print(index)
    #     # newlist[index] = alist[0]
    # with open('test2.txt', 'w') as f:
    #     print(newlist,file=f)
    # print(np.array(newlist))
    # print(np.random.choice([]))
    # b = np.load("test.npy",allow_pickle=True)
    # b2 = b.tolist()
    # with open('testadj2.txt', 'r') as f:
    #     mylist = literal_eval(f.read())
    #     print(len(mylist))
def getCompNodeCsv():#根据连通子图的节点，筛选node.csv中对应序号的行，选择最大的连通子图
    with open('componentNode.txt','r') as f:#记录旧序号，新索引和值的映射就是新索引和旧索引的映射
        nodes_to_keep = literal_eval(f.read())
    nodecsv = pd.read_csv('node.csv')
    edgecsv = pd.read_csv('edge.csv')#使用source target格式
    addrlist = nodecsv['Id']
    nodenum = 17339#是从原来的图中留下最大连通子图，非最大连通子图的信息不加载
    adjindex = np.arange(0, nodenum + 1)
    index2adj = dict(zip(adjindex, addrlist))#旧索引和地址的映射
    with open('oldindex2addr.txt','w') as f:
        print(index2adj,file=f)
    return
    addr_to_keep = [index2adj[index] for index in nodes_to_keep]
    CompNodeCsv = nodecsv[(nodecsv['Unnamed: 0'].isin(nodes_to_keep))]
    CompEdgeCsv = edgecsv[(edgecsv['Source']).isin(addr_to_keep) & (edgecsv['Target']).isin(addr_to_keep)]
    CompNodeCsv.to_csv('nodec.csv')
    CompEdgeCsv.to_csv('edgec.csv')

def attackedEdge():#将攻击后的邻接矩阵转换为地址source target类型的df
    with open('oldindex2addr.txt','r') as f:
        oldindex2addr = literal_eval(f.read())
    with open('componentNode.txt','r') as f:
        nodes_to_keep = literal_eval(f.read())#下标是新索引，值是旧索引
    # edgecsv = pd.read_csv('edge.csv')
    # source = edgecsv['Source']
    # target = edgecsv['Target']
    # weight = edgecsv['Weight']
    # sttuple = zip(source,target)
    # st2w = dict(zip(sttuple,weight))
    # with open('edgeDict.txt','w') as f:
    #     print(st2w,file=f)
    with open('edgeDict.txt','r') as f:
        st2w = literal_eval(f.read())
    # return
    csrmatrix = sp.load_npz('testmDice.npz')
    csrmatrix = sp.load_npz('testmEmbed.npz')#改变攻击方法，修改这里和下面按照文件名写入的一行即可
    csrmatrix = sp.load_npz('testmRandom.npz')
    coomatrix = sp.coo_matrix(csrmatrix)#row，col是非零值的新索引
    newrow = coomatrix.row#16295*16295不需要还原回全部的节点，只需要最大连通图
    newcol = coomatrix.col
    newindex = np.arange(len(nodes_to_keep))
    newindex2oldindex = dict(zip(newindex,nodes_to_keep))
    # df = pd.DataFrame.sparse.from_spmatrix(csrmatrix)
    # print(len(df[0]))
    # print(len(nodes_to_keep))
    # csrmatrix = sp.load_npz('testmEmbed.npz')
    # df = pd.DataFrame.sparse.from_spmatrix(csrmatrix)
    # print(len(df[0]))
    # csrmatrix = sp.load_npz('testmRandom.npz')
    # df = pd.DataFrame.sparse.from_spmatrix(csrmatrix)
    # print(len(df[0]))
    # csrmatrix = sp.load_npz('testmEmbed2.npz')
    # df = pd.DataFrame.sparse.from_spmatrix(csrmatrix)
    # print(len(df[0]))
    # return
    # with open('edgemD.csv','w',newline='') as f:
    # with open('edgemE.csv', 'w', newline='') as f:
    with open('edgemR.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Source','Target','Weight'])#要还原回原来的weight，如果原来没有这条边则设置为1
        for index in range(len(newrow)):
            rowi = newrow[index]
            coli = newcol[index]
            sourceOldIndex = newindex2oldindex[rowi]
            sourceAddr = oldindex2addr[sourceOldIndex]
            targetOldIndex = newindex2oldindex[coli]
            targetAddr = oldindex2addr[targetOldIndex]
            if (sourceAddr,targetAddr) in st2w.keys():
                weighti = st2w[(sourceAddr,targetAddr)]
                # print((sourceAddr,targetAddr))
                if weighti != 1:
                    print((sourceAddr, targetAddr))
            else:
                weighti = 1
                # print((sourceAddr,targetAddr))
                # return
            writer.writerow([sourceAddr,targetAddr,weighti])
    # df.to_csv('edgemD.csv')
    # print(type(df))
    # print(df.dtypes)
    # df = pd.read_csv('edge.csv')
    # print(type(df))
    # print(df.dtypes)

    # attackedAdj = sp.load_npz('testmEmbed.npz')
    # df = pd.DataFrame.sparse.from_spmatrix(attackedAdj)
    # df.to_csv('edgemE.csv')
    #
    # attackedAdj = sp.load_npz('testmEmbed2.npz')
    # df = pd.DataFrame.sparse.from_spmatrix(attackedAdj)
    # df.to_csv('edgemE2.csv')
    #
    # attackedAdj = sp.load_npz('testmRandom.npz')#testm表示modified
    # df = pd.DataFrame.sparse.from_spmatrix(attackedAdj)
    # df.to_csv('edgemR.csv')

def graphData():#将随机抽出的节点的特征填到node.csv中，unknown节点的特征值没有则置为0，应该补充和欺诈节点以及正常节点相连的未知节点的特征
    # mlnode = pandas.read_csv('mlnode.csv')#将每个节点的特征补充到正确的node.csv中
    mledge = pandas.read_csv('mledge.csv')
    # mlnode2 = pandas.read_csv('mlnode2.csv')
    mledge2 = pandas.read_csv('mledge2.csv')#挑出所有已知节点
    feature = pandas.read_csv('mlchar.csv')[["address","allIncome","allOutcome","avgIncome","avgOutcome","intxs","outtxs","livingTime","front1/3in","middle1/3in","last1/3in","front1/3out","middle1/3out","last1/3out","type"]]
    addrlist = list(feature["address"])
    for index,row in feature.iterrows():
        if row['type'] == 1:
            row['type'] = 'scam'
            feature.iloc[index] = row
        else:
            row['type'] = 'normal'
            feature.iloc[index] = row
    feature.rename(columns={'address': 'Id'},inplace=True)
    feature['degree'] = feature[['intxs', 'outtxs']].sum(axis=1)
    feature.to_csv('node.csv')#里面没有未知节点
    # mledge = mledge[(mledge['Source'].isin(addrlist)) & (mledge['Target'].isin(addrlist))]#选取两个节点都已知的边
    # mledge2 = mledge2[(mledge2['Source'].isin(addrlist)) & (mledge2['Target'].isin(addrlist))]
    # frames = [mledge,mledge2]
    # finaledge = pandas.concat(frames)
    # finaledge.to_csv('edge.csv')
def dprmatrix():#将用于pyg的转换为用于dpr攻击的
    edgecsv = pd.read_csv('edge.csv')#需要给每个节点一个编号，使用node.csv中的行号作为编号
    nodecsv = pd.read_csv('node.csv')
    nodenum = 17339
    attrnum = 14
    adjindex = np.arange(0,nodenum+1)
    attrindex = np.arange(0,attrnum+1)
    attrname = np.array(['allIncome','allOutcome','avgIncome','avgOutcome','intxs','outtxs','livingTime','front1/3in',
                         'middle1/3in','last1/3in','front1/3out','middle1/3out','last1/3out','degree'])
    addrlist = nodecsv['Id']
    adjdict = dict(zip(addrlist,adjindex))
    attrdict = dict(zip(attrname,attrindex))#属性名和属性索引的映射
    adjmatrix = np.zeros([nodenum,nodenum])
    attrmatrix = np.zeros([nodenum,attrnum])
    nodecsv.loc[nodecsv['type'] == 'scam', 'type'] = np.float16(1)
    nodecsv.loc[nodecsv['type'] == 'normal', 'type'] = np.float16(0)
    for index,row in edgecsv.iterrows():
        index1 = adjdict[row['Source']]
        index2 = adjdict[row['Target']]
        adjmatrix[index1][index2] = row['Weight']#转换为nodenum*nodenum大的邻接矩阵，有权矩阵
    class_names = np.array(['scam','normal'])
    adj_shape = np.array([nodenum,nodenum])
    attr_shape = np.array([nodenum,attrnum])
    for index,row in nodecsv.iterrows():
        rowindex = adjdict[row['Id']]
        for attr in attrname:
            colindex = attrdict[attr]
            attrmatrix[rowindex][colindex] = row[attr]
    adjcsr = sp.csr_matrix(adjmatrix)#压缩为稀疏矩阵，怀疑np.load会导致解压的npz文件的data全为0
    with np.load('test.npz') as loader:
        adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                             loader['adj_indptr']), shape=loader['adj_shape'])
    print(adj.data)
    return
    attrcsr = sp.csr_matrix(attrmatrix)
    adj_data = adjcsr.data
    np.set_printoptions(threshold=np.inf)
    print(adj_data)

    return
    adj_indices = adjcsr.indices
    adj_indptr = adjcsr.indptr
    adj_shape = adjcsr.shape
    attr_data = attrcsr.data
    attr_indices = attrcsr.indices
    attr_indptr = attrcsr.indptr
    attr_shape = attrcsr.shape
    format = 'csr'
    labels = nodecsv['type']
    labels = labels.astype(np.float32)

    # print(type(labels))
    # for label in labels:
    #     print(type(label))
    np.savez('test.npz', adj_data = adjcsr.data,adj_indices = adjcsr.indices,adj_indptr = adjcsr.indptr,adj_shape=adj_shape,
             attr_data = attrcsr.data,attr_indices = attrcsr.indices,attr_indptr = attrcsr.indptr,attr_shape = attrcsr.shape,
             class_names=class_names,labels = labels)#原始版本叫test.npz，选出一个最大连通子图component后的版本叫testc.npz
    # np.savez('test_meta_adj_0.05.npz', data=adjcsr.data, indices=adjcsr.indices, indptr=adjcsr.indptr,
    #          shape=adj_shape, attr_data=attrcsr.data, attr_indices=attrcsr.indices, attr_indptr=attrcsr.indptr, attr_shape=attrcsr.shape,
    #          class_names=class_names, labels=labels, format=format)
    np.savez('test_meta_adj_0.05.npz', data=adjcsr.data, indices=adjcsr.indices, indptr=adjcsr.indptr,
             shape=adj_shape, attr_data=attrcsr.data, attr_indices=attrcsr.indices, attr_indptr=attrcsr.indptr,
             attr_shape=attrcsr.shape, class_names=class_names, labels=labels, format=format)

    # npz = np.load('test_meta_adj_0.05.npz')
    # print(type(npz))
    # print(npz['labels'][0])
    # y = torch.from_numpy(npz['labels']).to(torch.float)
    # print(y)
    # print(csr)
def dprmatrixc():#将用于pyg的转换为用于dpr攻击的
    edgecsv = pd.read_csv('edgec.csv')#需要给每个节点一个编号，使用node.csv中的行号作为编号
    nodecsv = pd.read_csv('nodec.csv')
    nodenum = 16289
    attrnum = 14
    adjindex = np.arange(0,nodenum+1)
    attrindex = np.arange(0,attrnum+1)
    attrname = np.array(['allIncome','allOutcome','avgIncome','avgOutcome','intxs','outtxs','livingTime','front1/3in',
                         'middle1/3in','last1/3in','front1/3out','middle1/3out','last1/3out','degree'])
    addrlist = nodecsv['Id']
    adjdict = dict(zip(addrlist,adjindex))
    attrdict = dict(zip(attrname,attrindex))#属性名和属性索引的映射
    adjmatrix = np.zeros([nodenum,nodenum])
    attrmatrix = np.zeros([nodenum,attrnum])
    nodecsv.loc[nodecsv['type'] == 'scam', 'type'] = np.float16(1)
    nodecsv.loc[nodecsv['type'] == 'normal', 'type'] = np.float16(0)
    for index,row in edgecsv.iterrows():
        index1 = adjdict[row['Source']]
        index2 = adjdict[row['Target']]
        adjmatrix[index1][index2] = row['Weight']
    class_names = np.array(['scam','normal'])
    adj_shape = np.array([nodenum,nodenum])
    attr_shape = np.array([nodenum,attrnum])
    for index,row in nodecsv.iterrows():
        rowindex = adjdict[row['Id']]
        for attr in attrname:
            colindex = attrdict[attr]
            attrmatrix[rowindex][colindex] = row[attr]
    adjcsr = sp.csr_matrix(adjmatrix)
    attrcsr = sp.csr_matrix(attrmatrix)
    adj_data = adjcsr.data
    adj_indices = adjcsr.indices
    adj_indptr = adjcsr.indptr
    adj_shape = adjcsr.shape
    attr_data = attrcsr.data
    attr_indices = attrcsr.indices
    attr_indptr = attrcsr.indptr
    attr_shape = attrcsr.shape
    format = 'csr'
    labels = nodecsv['type']
    labels = labels.astype('int32')
    print(type(labels))
    # for label in labels:
    #     print(type(label))
    np.savez('testc.npz', adj_data = adjcsr.data,adj_indices = adjcsr.indices,adj_indptr = adjcsr.indptr,adj_shape=adj_shape,
             attr_data = attrcsr.data,attr_indices = attrcsr.indices,attr_indptr = attrcsr.indptr,attr_shape = attrcsr.shape,
             class_names=class_names,labels = labels)#原始版本叫test.npz，选出一个最大连通子图component后的版本叫testc.npz
    # np.savez('test_meta_adj_0.05.npz', data=adjcsr.data, indices=adjcsr.indices, indptr=adjcsr.indptr,
    #          shape=adj_shape, attr_data=attrcsr.data, attr_indices=attrcsr.indices, attr_indptr=attrcsr.indptr, attr_shape=attrcsr.shape,
    #          class_names=class_names, labels=labels, format=format)
    np.savez('testc_meta_adj_0.05.npz', data=adjcsr.data, indices=adjcsr.indices, indptr=adjcsr.indptr,
             shape=adj_shape, attr_data=attrcsr.data, attr_indices=attrcsr.indices, attr_indptr=attrcsr.indptr,
             attr_shape=attrcsr.shape, class_names=class_names, labels=labels, format=format)
def mymax():
    maxlist = []
    with open('scam2athirdLivingtime3.txt', 'r') as f:
        scamLivingtime = literal_eval(f.read())
    scamLivingtimeL = scamLivingtime.values()
    scamLivingtimeA,scamLivingtimeI = max(scamLivingtimeL),min(scamLivingtimeL)
    with open('scamIncome.txt','r',encoding='utf-8') as f:
        scamIncome = literal_eval(f.read())
    scamIncomeL = scamLivingtime.values()
    scamIncomeA, scamIncomeI = max(scamIncomeL), min(scamIncomeL)
    with open('scamOutcome.txt','r',encoding='utf-8') as f:
        scamOutcome = literal_eval(f.read())
    scamOutcomeL = scamOutcome.values()
    scamOutcomeA, scamOutcomeI = max(scamOutcomeL), min(scamOutcomeL)
    with open('scamInTx.txt','r') as f:#总的in tx num，但是命名不一致
        scamInTx = literal_eval(f.read())
    scamInTxL = scamOutcome.values()#L表示list，A表示max，I表示min
    scamInTxA, scamInTxI = max(scamInTxL), min(scamInTxL)
    with open('scamOutTx.txt','r') as f:
        scamOutTx = literal_eval(f.read())
    scamOutTxL = scamOutTx.values()
    scamOutTxA, scamOutTxI = max(scamOutTxL),min(scamOutTxL)

    with open('normal2athirdtime3.txt', 'r') as f:
        norLivingtime = literal_eval(f.read())
    norLivingtimeL = norLivingtime.values()
    norLivingtimeA,norLivingtimeI = max(norLivingtimeL),min(norLivingtimeL)
    with open('normalIncome.txt','r') as f:
        normalIncome = literal_eval(f.read())
    normalIncomeL = normalIncome.values()
    normalIncomeA,normalIncomeI = max(normalIncomeL),min(normalIncomeL)
    with open('normalOutcome.txt','r') as f:
        normalOutcome = literal_eval(f.read())
    normalOutcomeL = normalOutcome.values()
    normalOutcomeA, normalOutcomeI = max(normalOutcomeL), min(normalOutcomeL)
    with open('normal2athirdIntxnum.txt','r') as f:#其实是intx num
        normal2athirdIntxnum = literal_eval(f.read())
    normal2athirdIntxnumL = normal2athirdIntxnum.values()
    normal2athirdIntxnumA, normal2athirdIntxnumI = max(normal2athirdIntxnumL), min(normal2athirdIntxnumL)
    with open('normal2athirdOuttxnum.txt','r') as f:
        normal2athirdOuttxnum = literal_eval(f.read())
    normal2athirdOuttxnumL = normal2athirdOuttxnum.values()
    normal2athirdOuttxnumA, normal2athirdOuttxnumI = max(normal2athirdOuttxnumL), min(normal2athirdOuttxnumL)
    maxlist.append(scamLivingtimeA)
    maxlist.append(scamLivingtimeI)
    maxlist.append(scamIncomeA)
    maxlist.append(scamIncomeI)
    maxlist.append(scamOutcomeA)
    maxlist.append(scamOutcomeI)
    maxlist.append(scamInTxA)
    maxlist.append(scamInTxI)
    maxlist.append(scamOutTxA)
    maxlist.append(scamOutTxI)

    maxlist.append(norLivingtimeA)
    maxlist.append(norLivingtimeI)
    maxlist.append(normalIncomeA)
    maxlist.append(normalIncomeI)
    maxlist.append(normalOutcomeA)
    maxlist.append(normalOutcomeI)
    maxlist.append(normal2athirdIntxnumA)
    maxlist.append(normal2athirdIntxnumI)
    maxlist.append(normal2athirdOuttxnumA)
    maxlist.append(normal2athirdOuttxnumI)
    with open('maxlist.txt','w') as f:
        print(maxlist,file=f)
def dpr2pyg():#
    dpradj = np.load('testadj.npz')
    dprfeature = np.load('testfeature.npz')#将npz解压出转换为node.csv,edge.csv
    nodenum = 17339
    attrnum = 14
    print(type(dpradj))
    # np_to_csv = pd.DataFrame(data=dpradj)
    # np_to_csv.to_csv('test.csv')
    print(dpradj.files)
def scamvictim():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    print(len(addrlist))
    return
    df1 = pandas.read_csv('ntx.csv')
    df2 = pandas.read_csv('itx.csv')
    frames = [df1,df2]
    df = pandas.concat(frames)
    num = 0
    for index, row in df.iterrows():
        if isinstance(row['to'], str) and row['to'] in addrlist:
            num += 1
    print(num)#198407
def addressnum():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    with open('normalAddr.txt','r',encoding='utf-8') as f:
        addrlist2 = literal_eval(f.read())
    df1 = pandas.read_csv('ntx.csv')
    df2 = pandas.read_csv('itx.csv')
    frames = [df1, df2]
    print(df1.shape[0]+df2.shape[0])
    # df = pandas.concat(frames)
    # fromAddr = []
    # toAddr = []
    # for index, row in df.iterrows():
    #     if isinstance(row['from'],str):
    #         fromAddr.append((row['from']))
    #     if isinstance(row['to'],str):
    #         toAddr.append(row['to'])
    # fromAddr = list(set(fromAddr))
    # toAddr = list(set(toAddr))
    # print(len(fromAddr))#136064
    # print(len(toAddr))#40438
    # fromAddr = []
    # toAddr = []
    df1 = pandas.read_csv('normalAddrntx.csv')
    df2 = pandas.read_csv('normalAddritx.csv')
    print(df1.shape[0]+df2.shape[0])
    return
    frames = [df1, df2]
    df = pandas.concat(frames)
    for index, row in df1.iterrows():
        if isinstance(row['from'],str):
            fromAddr.append((row['from']))
        if isinstance(row['to'],str):
            toAddr.append(row['to'])
    for index, row in df2.iterrows():
        if isinstance(row['from'],str):
            fromAddr.append((row['from']))
        if isinstance(row['to'],str):
            toAddr.append(row['to'])
    fromAddr = list(set(fromAddr))#2541127
    toAddr = list(set(toAddr))#954942
    print(len(fromAddr))
    print(len(toAddr))
def util():#计算参数的占比
    with open('scamAvgIncome.txt', 'r', encoding='utf-8') as f:
        scamAvgIncome = literal_eval(f.read())
    with open('scamAvgOutcome.txt', 'r', encoding='utf-8') as f:
        scamAvgOutcome = literal_eval(f.read())
    with open('addr.txt', 'r', encoding='utf-8') as f:
        scamaddrlist = literal_eval(f.read())
    num1 = 0
    for addr, avgoutcome in scamAvgOutcome.items():
        if avgoutcome > 1:
            num1 += 1
    print(num1/len(scamaddrlist))

    with open('scamInTx.txt','r') as f:#总的in tx num，但是命名不一致
        scamInTx = literal_eval(f.read())
    with open('scamOutTx.txt','r') as f:
        scamOutTx = literal_eval(f.read())
    with open('addr.txt', 'r', encoding='utf-8') as f:
        scamaddrlist = literal_eval(f.read())
    # num1 = 0
    # for addr,intxnum in scamInTx.items():
    #     if intxnum > 100:
    #         num1 += 1
    # print(num1/len(scamaddrlist))


    with open('normalAvgIncome.txt','r') as f:
        normalAvgIncome = literal_eval(f.read())
    # num1 = 0
    # for addr,avgincome in normalAvgIncome.items():
    #     if avgincome > 1:
    #         num1 += 1
    # print(num1/len(scamaddrlist))
    with open('normalAvgOutcome.txt','r') as f:
        normalAvgOutcome = literal_eval(f.read())
    # num1 = 0
    # for addr, avgoutcome in normalAvgOutcome.items():
    #     if avgoutcome > 1:
    #         num1 += 1
    # print(num1 / len(scamaddrlist))
    with open('normal2athirdIntxnum.txt','r') as f:#其实是intx num
        normal2athirdIntxnum = literal_eval(f.read())
    with open('normal2athirdOuttxnum.txt','r') as f:
        normal2athirdOuttxnum = literal_eval(f.read())
    # num1 = 0
    # for addr,intxnum in normal2athirdIntxnum.items():
    #     if intxnum > 100:
    #         num1 += 1
    # print(num1/len(scamaddrlist))
    with open('fig15addr2livinghours.txt', 'r', encoding='utf-8') as f:#原来用nx的分组统计的living time还可以用
        fig15addr2livinghours = literal_eval(f.read())
    zipped = zip(fig15addr2livinghours.keys(), fig15addr2livinghours.values())
    sort_zipped = sorted(zipped, key=lambda x: (x[1]))
    print(sort_zipped)
def fig12expansion():
    df1 = pandas.read_csv('ntx.csv')
    df2 = pandas.read_csv('itx.csv')
    # addrlist = ['0x969244f550a683e6e3425c7cb87aa6afe5fda253','0x51af777899f0e81fbb69836e9255cc5bab7a5842','0x4bf722014e54aeab05fcf1519e6e4c0c3f742e43',
    #             '0x1fd41f2da3fcd5e61c4a6b70d387e480b6c38dba','0x83053c32b7819f420dcfed2d218335fe430fe3b5']
    classlist = [51,193,81,65,37]
    exp = {}
    for c in classlist:
        exp[c] = []
        filename = 'fig12-' + str(c) + '.csv'
        df = pd.read_csv(filename)
        addrlist = list(df['Id'])
        for addr in addrlist:
            for index, row in df1.iterrows():
                if isinstance(row['from'], str) and row['from'] == addr and isinstance(row['to'], str):
                    exp[c].append(row['to'])
                if isinstance(row['to'], str) and row['to'] == addr and isinstance(row['from'], str):
                    exp[c].append(row['from'])
            for index, row in df2.iterrows():
                if isinstance(row['from'], str) and row['from'] == addr and isinstance(row['to'], str):
                    exp[c].append(row['to'])
                if isinstance(row['to'], str) and row['to'] == addr and isinstance(row['from'], str):
                    exp[c].append(row['from'])
    with open('fig12exp.txt','w') as f:
        print(exp,file=f)
    for c in classlist:
        print(c)
        # exp[addr] = list(set(exp[addr]))
        print(len(exp[c]))
def txsimplify():
    df1 = pandas.read_csv('ntx.csv')
    df2 = pandas.read_csv('itx.csv')
    #只取from和to
    df1 = df1[['from','to']]
    df2 = df2[['from','to']]
    # print(df1)
    df1.to_csv('ntxFromTo.csv')
    df2.to_csv('itxFromTo.csv')
def fig12expansion2():
    df1 = pandas.read_csv('ntx.csv')
    df2 = pandas.read_csv('itx.csv')
    # addrlist = ['0x969244f550a683e6e3425c7cb87aa6afe5fda253','0x51af777899f0e81fbb69836e9255cc5bab7a5842','0x4bf722014e54aeab05fcf1519e6e4c0c3f742e43',
    #             '0x1fd41f2da3fcd5e61c4a6b70d387e480b6c38dba','0x83053c32b7819f420dcfed2d218335fe430fe3b5']
    classlist = np.arange(397)
    df3 = pandas.read_csv('fig12.csv')
    # print(classlist)
    # return
    exp = {}
    for c in classlist:
        exp[c] = []
        # filename = 'fig12-' + str(c) + '.csv'
        # df = pd.read_csv(filename)
        # addrlist = list(df['Id'])
        addrlist = df3[df3['class'] == str(c)]#搜索每一类的地址
        for addr in addrlist:#
            for index, row in df1.iterrows():
                if isinstance(row['from'], str) and row['from'] == addr and isinstance(row['to'], str):#找出每一类地址的一跳地址
                    exp[c].append(row['to'])
                if isinstance(row['to'], str) and row['to'] == addr and isinstance(row['from'], str):
                    exp[c].append(row['from'])
            for index, row in df2.iterrows():
                if isinstance(row['from'], str) and row['from'] == addr and isinstance(row['to'], str):
                    exp[c].append(row['to'])
                if isinstance(row['to'], str) and row['to'] == addr and isinstance(row['from'], str):
                    exp[c].append(row['from'])
    with open('fig12exp2.txt','w') as f:
        print(exp,file=f)
    for c in classlist:
        print(c)
        # exp[addr] = list(set(exp[addr]))
        print(len(exp[c]))
def fig12expansion2FromTo():#根据已有的gephi分类，通过欺诈交易统计每个地址的交易对象
    df1 = pandas.read_csv('ntxFromTo.csv')
    df2 = pandas.read_csv('itxFromTo.csv')
    # addrlist = ['0x969244f550a683e6e3425c7cb87aa6afe5fda253','0x51af777899f0e81fbb69836e9255cc5bab7a5842','0x4bf722014e54aeab05fcf1519e6e4c0c3f742e43',
    #             '0x1fd41f2da3fcd5e61c4a6b70d387e480b6c38dba','0x83053c32b7819f420dcfed2d218335fe430fe3b5']
    classlist = np.arange(0,397)
    df3 = pandas.read_csv('fig12.csv')[['Id','class']]
    # print(classlist)
    # return
    exp = {}
    # for c in classlist:
    #     exp[c] = []
    #     # filename = 'fig12-' + str(c) + '.csv'
    #     # df = pd.read_csv(filename)
    #     # addrlist = list(df['Id'])
    #     # print(c)
    #     mydf3 = df3[df3['class'] == c]#得出一类的地址，搜索每一类的地址的一跳
    #     addrlist = list(mydf3['Id'])
    #     # print(addrlist)
    #     # print(type(addrlist))
    #     for addr in addrlist:#addr不是
    #         # print('t')
    #         # print(addr)
    #         for index, row in df1.iterrows():
    #             if isinstance(row['from'], str) and row['from'] == addr and isinstance(row['to'], str):#找出每一类地址的一跳地址
    #                 exp[c].append(row['to'])
    #                 # print(row['to'])
    #             if isinstance(row['to'], str) and row['to'] == addr and isinstance(row['from'], str):
    #                 exp[c].append(row['from'])
    #                 # print(row['from'])
    #         for index, row in df2.iterrows():
    #             if isinstance(row['from'], str) and row['from'] == addr and isinstance(row['to'], str):
    #                 exp[c].append(row['to'])
    #                 # print(row['to'])
    #             if isinstance(row['to'], str) and row['to'] == addr and isinstance(row['from'], str):
    #                 exp[c].append(row['from'])
                    # print(row['from'])
    #先统计每个地址的交易地址，然后再将每个地址的交易地址分组,降低时间复杂度
    scamaddr2addr = defaultdict(list)
    for index, row in df1.iterrows():
        if isinstance(row['from'], str) and isinstance(row['to'], str):
            scamaddr2addr[row['from']].append(row['to'])
            scamaddr2addr[row['to']].append(row['from'])
    for index, row in df2.iterrows():
        if isinstance(row['from'], str) and isinstance(row['to'], str):
            scamaddr2addr[row['from']].append(row['to'])
            scamaddr2addr[row['to']].append(row['from'])
    scamaddr2addr = dict(scamaddr2addr)
    with open('scamaddr2addr.txt','w') as f:
        print(scamaddr2addr,file=f)
    # for c in classlist:
    #     exp[c] = []
    #     mydf3 = df3[df3['class'] == c]  # 得出一类的地址，搜索每一类的地址的一跳
    #     addrlist = list(mydf3['Id'])
    #     for addr in addrlist:
    #         exp[c] += scamaddr2addr[addr]
    # with open('fig12exp2FromTo.txt','w') as f:
    #     print(exp,file=f)
    with open('fig12exp2FromTo.txt','r') as f:#欺诈地址相关交易的每个地址的交易对象
        exp = literal_eval(f.read())#每个组拓展后的地址
    for c in classlist:
        print(str(c)+':')
        # print(':')
        # exp[addr] = list(set(exp[addr]))
        print(len(exp[c]))
def expansion():
    with open('fig12exp2FromTo.txt','r') as f:#算出每组拓展前后的区别
        exp = literal_eval(f.read())
    with open('gephiGroup.txt','r') as f:
        groupdict = literal_eval(f.read())
    differ = {}
    for index in range(0,397):
        differ[index] = len(exp[index]) - len(groupdict[index])
    differlist = list(differ.values())
    maxdiffer = max(differlist)
    groupindex = differlist.index(maxdiffer)
    print(groupindex)#51
    print(len(exp[groupindex]))#15712
    print(len(groupdict[groupindex]))#708
    print(maxdiffer)#15004
    maxgroup = groupdict[groupindex]
    maxgrouppartner = {}
    with open('scamaddr2addr.txt', 'r') as f:
        scamaddr2addr = literal_eval(f.read())
    df = pd.read_csv('scamaddr2addr.csv')
    for addr in maxgroup:
        maxgrouppartner[addr] = len(scamaddr2addr[addr])
    zipped1 = zip(maxgrouppartner.keys(),maxgrouppartner.values())
    maxgrouppartner = sorted(zipped1,key = lambda x:(x[1]))
    print(maxgrouppartner)
#找出第51组的最多交易对象的地址，先找51组所有地址，然后在哪里中找51组所有的交易对象
#51组全部是欺诈地址
#利润超过100eth的组的数量
#利润最高的组和它的利润，该组的平均利润
#所有组的受害者
#所有组最多受害者的组和受害者数量
def unknownNodeNtx():#未知地址已经被爬取的交易，但是这些交易只是和欺诈地址有关的交易
    # mlnodelist = pandas.read_csv('mlnode.csv')['Id']
    # with open('addr.txt', 'r', encoding='utf-8') as f:
    #     scamaddrlist = literal_eval(f.read())
    # with open('normalAddr.txt','r',encoding='utf-8') as f:
    #     noraddrlist = literal_eval(f.read())
    # unknownNodelist = [addr for addr in mlnodelist if addr not in scamaddrlist and addr not in noraddrlist]
    # with open('unknownAddr.txt','w') as f:
    #     print(unknownNodelist,file=f)
    with open('unknownAddr.txt','r') as f:
        unknownNodelist = literal_eval(f.read())
    print(len(unknownNodelist))
    return
    session = requests.Session()
    with open('unknownNtx.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["blockNumber", "timeStamp", "hash", "nonce", "blockHash", "transactionIndex",
                         "from", "to", "value", "gas", "gasPrice", "isError", "txreceipt_status", "input",
                         "contractAddress", "cumulativeGasUsed", "gasUsed", "confirmations"])
        for addr in unknownNodelist:
            url = "https://api.etherscan.io/api?module=account&action=txlist&address=" + addr + "&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey=" + apikey
            results = literal_eval(session.get(url).text)['result']
            try:
                if len(results):
                    for result in results:
                        writer.writerow([result['blockNumber'], result['timeStamp'], result['hash'], result['nonce'],
                                         result['blockHash'],
                                         result['transactionIndex'], result['from'], result['to'], result['value'],
                                         result['gas'], result['gasPrice'],
                                         result['isError'], result['txreceipt_status'], result['input'],
                                         result['contractAddress'], result['cumulativeGasUsed'],
                                         result['gasUsed'], result['confirmations']])
            except Exception:
                print(Exception)
                print(url)
                print(result)
                print(addr)
def es2csv():
    #读取整个文件夹的文件，不需要使用endswith
    filedir = r'D:\all_addr_txs'
    filelist = os.listdir(filedir)
    # filelist = [filelist[0]]
    #是一个dict，交易哈希，value是起点终点转账数额，时间戳，内部交易
    toaddr2value = {}#根据to地址得到地址对应的转账数目，不需要记录交易哈希
    # toaddr2victim = defaultdict(list)#统计未知地址的交易对象数量
    toaddr2victimNum = defaultdict(int)
    with open('unknownAddrtxs.csv','w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['from','to','value'])
        for filename in filelist:
            filename = filedir + '\\' + filename
            with open(filename, 'r') as f:
                try:
                    data = json.load(f)
                    for txhash,mylist in data.items():#转账数额是列表第三个，将交易对象统计到csv文件中
                        # toaddr2value[mylist[1]] = toaddr2value.get(mylist[1],0) + int(mylist[2]) / 1000000000000000000
                        # toaddr2value[mylist[1]] = toaddr2value.get(mylist[1],0) + int(mylist[2]) / 1000000000000000000
                        # toaddr2victim[mylist[1]].append(mylist[0])
                        writer.writerow([mylist[0],mylist[1],mylist[2]])
                        toaddr2victimNum[mylist[1]] += 1#不能根据交易的次数进行统计，可能有的地址交易多次
                except Exception:
                    traceback.print_exc()
    # with open('unknownAddrProfit.txt','w') as f:#还需要知道和未知地址交易的地址，用于计算未知地址被判断为欺诈地址后的受害者数量
    #     print(toaddr2value,file=f)
    # with open('unknownAddrVictim.txt', 'w') as f:
    #     print(toaddr2victim,file=f)
    with open('unknownAddrVictimNum.txt', 'w') as f:#错误的受害者求法
        print(toaddr2victimNum,file=f)
def unknownVictim():
    # df = pd.read_csv('unknownAddrtxs.csv',low_memory=False)#只有from，to，value
    with open('fig12exp2FromTo.txt','r') as f:#拓展后的组
        fig12exp2FromTo = literal_eval(f.read())
    #先统计受害者还是先去除同组地址，根据是否是同组地址选择是否统计
    #记录当前地址所在组，查询交易对象是否在同组即可知道是否应该统计
    #筛选df中交易对象为该地址的交易，然后判断交易对象
    #根据交易的from和to去重
    # df = df.drop_duplicates(subset=['from','to'])
    # df.to_csv('unknownAddrtxsFromTo.csv')
    df = pd.read_csv('unknownAddrtxsFromTo.csv',low_memory=False)
    # print(df.shape[0])
    addr2victim = defaultdict(int)
    # for index,addrlist in fig12exp2FromTo.items():
    #     for addr in addrlist:
    #         conflictSet = fig12exp2FromTo[index]
    #         txlist = df[(df['to'].notnull()) & (df['to'] == addr) & (df['from'].notnull()) & (~df['from'].isin(conflictSet))]
    #         addr2victim[addr] = txlist.shape[0]#
    df2 = list(pandas.read_csv('fig12.csv')[['Id']])
    with open('addr2expgroup.txt','r') as f:#拓展后的组
        addr2expgroup = literal_eval(f.read())
    print(0)
    for index,row in df.iterrows():#统计拓展后未知地址的受害者，未知地址的受害者可能是拓展后同一个分组的，可能不是拓展后的分组
        print(2)
        if isinstance(row['to'],str) and isinstance(row['from'],str):
            print(1)
            if row['from'] not in df2 and row['to'] in df2:
                print(3)
                addr2victim[row['to']] += 1
            if row['from'] in addr2expgroup.keys():
                print(4)
                groupindex = addr2expgroup[row['from']]
                if row['to'] in df2 and row['from'] in df2 and row['from'] not in fig12exp2FromTo[groupindex]:#fig12exp2FromTo是根据欺诈交易得到的分组，不是根据未知地址相关交易的
                    addr2victim[row['to']] += 1
                    print(5)
    addr2victim = dict(addr2victim)
    print(addr2victim)
    with open('unknownVictim.txt','w') as f:
        print(addr2victim,file=f)
def addr2expgroup():
    with open('fig12exp2FromTo.txt','r') as f:#拓展后的组
        fig12exp2FromTo = literal_eval(f.read())
    addr2group = defaultdict(int)
    for index,addrlist in fig12exp2FromTo.items():
        for addr in addrlist:
            addr2group[addr] = index
    addr2group = dict(addr2group)
    with open('addr2expgroup.txt','w') as f:
        print(addr2group,file=f)
import pdb
def tweetcrawler():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    addrlist = ['0xd0e929ea70a916e53b4606d7ea5280cb0ddaf7d1','0x53e00c6a2887f71bed5340ce369675ddaff4f42a','0x032fb380d84917d408ea172383cf214c28b4fe21']
    session = requests.Session()
    classname = 'css-1dbjc4n'
    xpath = '//*[@id="id__i76dt803nbg"]'
    noresult = '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div/div/div[2]/div/div/div/div[2]/text()[1]'
    retry = '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div/div/div[2]/div/div/div[1]/span'
    chromeOptions = webdriver.ChromeOptions()
    # 设置代理
    chromeOptions.add_argument("--proxy-server=https://127.0.0.1:1080")
    chromeOptions.add_argument("--proxy-server=https://127.0.0.1:1080")
    chromeOptions.add_argument("--proxy-server=socks5://127.0.0.1:1080")
    # chromeOptions.add_argument("--proxy-server=socks4://127.0.0.1:1080")
    # 一定要注意，=两边不能有空格，不能是这样--proxy-server = http://202.20.16.82:10152
    with webdriver.Chrome('chromedriver',chrome_options=chromeOptions) as w:#只打开一个浏览器
        for addr in addrlist:
            url = "https://twitter.com/search?f=top&q=" + addr + "&src=typed_query"
            w.get(url)
            time.sleep(10)
            # pdb.set_trace()
            try:
                if w.find_element_by_xpath(retry):#多次跑发现第一次必定需要刷新
                    w.refresh()
                    time.sleep(10)
                    try:
                        if w.find_element_by_xpath(retry):
                            print(addr)
                            addrindex = addrlist.index(addr)
                            print(addrindex)
                            return
                    except Exception:
                        # print(addr)
                        # addrindex = addrlist.index(addr)
                        # print(addrindex)
                        pass
            except Exception:
                pass
            try:
                if w.find_element_by_xpath(noresult):
                    continue
            except Exception:#如果异常则有结果，则用下一个if保存结果
                pass
            try:
                if w.find_element_by_xpath(xpath):#不会触发else，因为会直接触发异常
                    link = w.find_element_by_xpath(xpath)
                    url = "https://twitter.com/" + link#进入推特详情页，应该不止一个link
                    w.get(url)
                    time.sleep(10)
                # bs = BeautifulSoup(html, features="lxml")
                    twtimeXpath = '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/section/div/div/div[12]/div/div[1]/article/div/div/div/div[3]/div[3]/div/div[1]/a[1]/span'
                    twtime = w.find_element_by_xpath(twtimeXpath)
                    w.back()  # 返回搜索结果页，然后读取下一个推特内容
                    print(twtime)
            except Exception:#前面已经处理了两种异常
                print(addr)
                addrindex = addrlist.index(addr)
                print(addrindex)
                traceback.print_exc()
                # continue
            #     pass
#之前因为只使用requests被反爬了
def tweetcsv():
    t1 = pd.read_csv('t.csv')
    t2 = pd.read_csv('t2.csv')
    t3 = pd.read_csv('t3.csv')
    t4 = pd.read_csv('t4.csv')
    t5 = pd.read_csv('t5.csv')
    t6 = pd.read_csv('t6.csv')
    t7 = pd.read_csv('t7.csv')
    t8 = pd.read_csv('t8.csv')
    # print(t1.shape[0])
    # frames = [t1,t2,t3,t4,t5,t6,t7,t8]
    # df = pd.concat(frames)
    # df.to_csv('scamTweet.csv')
    df = pd.read_csv('scamTweet.csv')
    print(df.shape[0])#954
    re = '/(^|[^@\w])@(\w{1,15})\b/\sˆ0x[a-fA-F0-9]40$'#匹配用户名和以太坊地址,还是手动筛选检查
    for row,index in df.iterrows():
        tweet = row['tweet']
#604个欺诈报告，100个引诱打钱，28个非英语，111+111=222个只有用户名和地址
def website():
    with open('addr.txt','r',encoding='utf-8') as f:
        addrlist = literal_eval(f.read())
    testaddr = []#需要手动筛选关键词的地址列表
    addr2tag = defaultdict()#表示所有地址的关键词情况
    # addrlist = ['0xf832a212d1c81b89e2dabc79c728e6a294ea5edd']
    with open('website.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["addr", "tagline"])
        for addr in addrlist:
            try:
                filename = r'D:\chrome_download\Search-Engines-Scraper-master\mycsv' + '\\' + addr + '.csv'
                keywords = ['scam','fraud','blacklist','etherscan','breadcrumb','explorer','ethplorer','bscscan','xplorer','blockchain.com',
                            '0xzx.com','app.','zerion','wallet','uniswap123','eth1.bcfn','kycaml.yezcoin','dex.guru','coingecko','ack.pl',
                            'github.com','eth.tokenview','coinmarketcap','blockchain.info','cn.vechainstats','/account/','/address/',
                            'stackoverflow','saoniuhuo.com','minergate.com','gitlab','dokumen.tips','bitcoinabuse','github','blockchair',
                            'icy.tools','bitcointalk','abc.bi','/contracts/','arxiv','tradeblock','pure.rug.nl','rs.booksc.org','coincarp.com',
                            'coinpan','etherchain','ethereum.stackexchange','nomics.com','coinvietnam.vn','pancakeswap.finance','hack',
                            'recentcoin.com','conspi.ru','certik.com','/accounts/','tokenview.com','token-profile.token','urlscan.io','phish',
                            'thebittimes.com','polygonscan.com','stole','books.google.com','pdfcoffee.com','scribd.com','x.blockeye','thebittimes.com',
                            'ftmscan.com','ethvm.com','fraud','/token/','rekt.news','coincheck.blog','reentrancy','attack','mobile.twitter.com',
                            'groups.google.com','blockglobe24.com','ymsmhj.com','panewslab.com','btcfans.com','spamreports.report','accesspegasus.com',
                            'jinse.com','骗','黑客','騙','监测','盗']#需要排除的关键字
                df = pd.read_csv(filename)
                import pdb
                # pdb.set_trace()
                #某个地址csv的所有行都不包含关键字的时候打印该地址
                testaddrrow = []
                addrtag = [True] * df.shape[0]  # 检测地址csv每行是否含有关键词，长度是csv文件长度
                df = df[26:]#df的序号和csv用excel打开的行号的关系：序号=行号-2
                for index,row in df.iterrows():
                    try:
                        domain = row['domain'].lower()
                        URL = row['URL'].lower()
                        title = row['title'].lower()
                        if isinstance(row['text'], str):
                            text = row['text'].lower()
                        elif not isinstance(row['text'], str):
                            text = ''
                        # for keyword in keywords:#遍历每个关键词改成用all方法
                        # rowindex = keywords.index(keyword)#检测行是否含有关键词
                        domaintag = all([keyword not in domain.lower() for keyword in keywords])#domain域不包含任何一个关键词tag才为True，含有任何一个关键词则为False
                        URLtag = all([keyword not in URL.lower() for keyword in keywords])
                        titletag = all([keyword not in title.lower() for keyword in keywords])
                        texttag = all([keyword not in text.lower() for keyword in keywords])
                        import pdb
                        # pdb.set_trace()
                        #addrtag表示某个地址的行关键词情况，长度是csv文件的长度
                        #rowtag表示某个地址某行的关键词情况，长度是4
                        #addr2tag表示地址列表的关键词情况，长度是地址列表长度
                        #testaddrrow表示地址中特殊的关键词行，作为addr2tag的值，和addrtag作用类似，addrtag记录某个地址所有的行情况，testaddrrow记录特殊的行索引
                        #testaddr表示所有地址的关键词行
                        rowtag = [domaintag, URLtag, titletag, texttag]#域名tag=True表示含有关键词，=0False表示不含任何一个关键词，该行的关键词状况
                        if any([t == False for t in rowtag]):#每个tag都是0False表示该行所有域都不含关键词，该行任何一个域存在关键词则不记录
                            addrtag[index] = False#该行含关键词，需要记录都不含关键词的行方便手动提取关键词，打包地址和不含关键词的行索引到testaddr中，用列表打包
                        elif all([t == True for t in rowtag]):#含有任何一个关键词t=False则不记录，都不含有关键词才记录
                            testaddrrow.append(index)#该行任何一个域含有关键词则记录该行号
                            addrtag[index] = True#True表示该行需要记录，该行都不含关键词
                    except:
                        import traceback
                        traceback.print_exc()
            except:
                import traceback
                traceback.print_exc()
            if any([t == True for t in addrtag]):#处理完一个地址后，某一行没有关键词则打印地址和index
                testaddr.append(testaddrrow)#
            writer.writerow([addr, testaddrrow])#处理完一个地址后输出csv
            addr2tag[addr] = testaddrrow
    # addr2tag = dict(addr2tag)
    # with open('website.txt', 'w') as f:
    #     print(testaddr,file=f)
    # with open('webaddr2tag.txt', 'w') as f:
    #     print(addr2tag,file=f)
    # print(addr2tag)
    #得出的csv很多重复，不是重启搜索引擎程序出现问题，爬虫本身有问题
#文件：0x23cb06dc044f4c9937866a5928c2427639f390a0 0xbb835bd2259d34e4ea77ef39ba63ec1d93a047e0 0x7d52327b12afaead317552a1e96bc55eb8e0e52c 0x479cc10103b079ace9e9bb2db1f81cdcd5cdf642 0x63f32b8ec6e4dea892149834a7b8d7c9e67b4a0a
# 0x938fdac116de11faf8e682e93dec3c849895e380 0xdba8f7ba4be466e03bc7d7c6e86f95b36da1b642 0x54189eab44661be1b081b4535c050230a787d816 0xdf8ee0a230894e7e512dc3e1c9f5df520bd6b84e 0x72dc6ad43749a9fa2ae381ed35b5fd8f6604c7bf
# 0x3715b1ed9bbd5e165af97619b5c5e13e39f55504 0xe7eb3ddff9591915e65f06bc4cda8eee0224e461 0x92132d45119fae2f375db1cb835bb4ec46a9e71c 0x4ac6307a85d83962503f86457de9c331a6926f48 0xeeeeeeeeb3086c7b26b2027f1d42e83c36d49489
# 0xbb7ef8418e9a74c0c731aaaab688aabc9b7af857 0xc0b9c8dcbabbe0f0febff1a823d87615392aa844 0x8d18b40c3b095f2d8ea99bf3a814c7182e2f8efd 0x4c8093c75fc778008d96f865f770233d6acf724e
# 0x6350fb4a8bc3cda203cfe00f87ba0f7b3bf61600
#手动筛选：
#https://teletype.in/@ethereum2.0/_7Slcqiy
#https://tgraph.io/Ethereum-20-News-03-04
#https://medium.com/@RiseICO/crowdsales-is-live-now-c01de3db4d09
#https://www.sweety.cn/175200/%E4%BB%A3%E5%B8%81%E9%94%80%E5%94%AE%E6%AD%A3%E5%9C%A8%E8%BF%9B%E8%A1%8C%E4%B8%AD.html 无
#https://top100token.com/newborn/0x003Eb9C77B5B896fcC27aDEAd606d23DeF34510e 无
#https://coinpan.com/free/108934444 分析
#https://thebittimes.com/token-VB-ETH-0x1d755f741377ddb79ba7efa5beaf0305672e6707.html
#https://7asabco.org/showthread.php?t=52496
#https://gafsa.jeun.fr/t45505-topic
#https://impumelelo.net/site/c3f67f-callisto-network-twitter 无
#https://medium.com/@cryptolasi/callisto-network-d1b510e2f709
#https://blog.kakaocdn.net/dn/bO3hG0/btqtY8oDEoi/U6YpW01XgbcZrqX1S8Bya1/20190330칼리스토 재정보고서.pdf?attach=1&knm=tfile.pdf
#https://www.scribd.com/document/565498341/Callisto-Network-WP-1-6-1
#https://ww2.easycars24.pl/de-lu/895388637.php 无
#https://steemit.com/ico/@oxygen77/obzor-proekta-etheera
#https://thebitcoinstreetjournal.com/ethereum-ceo-vitalik-buterin-eth-giveaway-2/
#https://telemetr.io/en/channels/1273640839-yolkfinancechannel 无
#https://nenettegarcia.com/jpr1u7v/who-is-the-publisher/smart-contract-address-ethereum.html
#https://tgstat.com/channel/@hotairdropper
#https://www.coinlore.com/coin/void
#https://www.coincarp.com/currencies/fuzetoken/richlist/
#https://medium.com/@thursdayninjatoken/how-to-claim-your-thur-airdrop-phase-1-c888af0bfe80
#https://archived.moe/biz/thread/14585469
#https://m.facebook.com/devolperandroid/photos/a.698660063926775/757322908060490/?type=3&source=54
#https://t.me/s/Crypto_Earn_BoT1?before=1964
#https://steemkr.com/kr/@ai1love/2tkorr 不知道是不是
#http://m.maybugs.com/news/articleView.html?idxno=662702 不知道是不是
#https://t.me/s/legitetheriumsales?before=25
#https://telegra.ph/Ethereum-20-Foundation-100000-ETH-Giveaway-Airdrop-07-06
#http://119.29.176.244:8001/position/tomatosfinance/7day
#https://m.facebook.com/pg/Crypter%C3%ACum-112903863757045/posts/
#https://t.me/s/EthereumPrivateChannel
#https://peakd.com/itismycoinblog/@m0nika66/ethereum-private-ethpt-secured-and-decentralized-innovative-cryptocurrrency
#https://eslilin854645586.wordpress.com/page/8/ 无
#https://medium.com/matic-network-official/matic-network-token-sale-launch-3ec86fc3d46
#https://www.coinlore.com/coin/giftedhands
#https://www.coinlore.com/coin/crad-cash
#https://medium.com/@torchain.io/new-bonus-added-586ae05ab602
#https://telegra.ph/UNI-Foundation-5000000-UNISWAP-Giveaway-09-18
#https://beta.peakd.com/@ashwim/re-techguru-re-ashwim-experty-ico-hacked-more-than-usd150-000-stolen-20180130t042403564z
#https://forum.gipsyteam.ru/index.php?viewtopic=132721&st=80 无
#https://www.coinlore.com/coin/giga-watt-token
#https://medium.com/@toktz244?source=rss-25b17f3bb816------2
#https://gitter.im/kovan-testnet/faucet?at=5addb0656d7e07082b237ded
#https://gitter.im/ethereum/go-ethereum/archives/2017/09/20
#https://telegra.ph/Ethereum-Foundation-100000-ETH-Giveaway-Airdrop-11-22
#https://id.linkedin.com/pulse/petchains-bunga-tulip
#https://bitcoinf.forum2x2.ru/t390-topic
#https://telegram-store.com/catalog/channels/medibeu/80
#https://tgstat.com/channel/@Medibeu
#https://www.truegoldcoin.com/
#https://telegra.ph/Huobi-04-13
#https://ethgiveaway.droppages.com/
#https://ethereumdev.medium.com/ethereum-2-0-deposit-contract-871a1b32ecfb
#https://telegra.ph/Huo-03-12 借着科普的名义把欺诈地址嵌入到文章中，火币，好币
#https://telegra.ph/HT-03-11
#https://mobile.twitter.com/nftglhm
#https://www.haobi8.com/biwd/25101/
#https://t.me/s/powerpoolcvp?before=129
#https://id.linkedin.com/pulse/petchains-bunga-tulip
#https://www.ptcs.io/
#https://payment-address-coins.netlify.app/
#http://rafajunior99.blogspot.com/2018/05/hashcard-ico-hshc-token-crypto-to-fiat.html
#https://t.me/s/tradecryptonow?before=1031
#https://telegra.ph/ethereum-foundation-airdrop-eth-02-12-2
#https://www.coinlore.com/pl/coin/kimex
#https://smart-car.tech/business-essentials/paywithbitcoin.html 收款二维码
#https://seekoin.com/addr-14aDScg4SeF4GDNLTpXezGPsfKkhoENweV
#https://telegra.ph/HT-09-14
#https://medium.com/@HannahD11802153/vechain-foundation-airdrop-ff921b73d187
#http://119.29.176.244:8001/position/kappitoken 无
#https://t.me/s/HBC_announcement?before=22
#https://www.jeuxvideo.com/forums/42-3011927-64524787-1-0-1-0-airdrop-x-stable-coin.htm airdrop
#https://m.facebook.com/Diamond-Project-156763648321824/?ref=page_internal ico
#https://tgstat.com/channel/@icointel
#https://telemetr.io/uk/channels/1092338095-icointel
#https://www.icoannouncement.io/listing/yearn-hold-finance-yhfi/ presale
#https://www.haobi8.com/biwd/29982/ 科普
#https://tgstat.com/channel/@icointel
#https://telemetr.io/uk/channels/1092338095-icointel
#https://www.norkerens.es/cu/136gv83220h/ 无
#https://asd.lucianomanzalini.it/ho/391gv11901x/ 无
#https://huriueigremoer.xyz/do/638ts27210u/ 无
#https://www1.melipop.es/xu/891hg121302z/ 无
#https://www.coinlore.com/coin/iungo
#http://119.29.176.244:8001/position/iungo/7day
#https://t.me/s/HYIPcontract tele
#https://nftrade.com/users/bsc/0x776d5986b34304984f0263c8c718ce73a0515d15?search=&sort=listed_desc nft相关
#http://119.29.176.244:8001/position/pumpfarm/7day 无
#https://forum.bitcoin.com/ethereum/dogcash-dog-airdrop-round-1-t117211.html airdrop
#https://forumbitcoin.co.id/threads/dogcash-dog-airdrop-round-1.101021/
#https://sw-ke.facebook.com/TheMasterofCrypto1/posts/343655919623054 外
#http://www.qb03.com/news/show-2244.html 无法访问
#http://smeasp.com/shipu/yaowen/13299.html 无法访问
#https://sx.qinyuequan.com/forum.php?mod=viewthread&tid=178540&mobile=2 无
#https://www.coinlore.com/coin/yang
#https://nitter.domain.glass/WePowerN/status/967415097969774592 无
#https://www.tokenpost.kr/article-25900 外
#https://cobak.co.kr/community/1/post/296007 外
#https://tgstat.com/ru/channel/@airdropycckux 外
#https://www.globees.co.uk/contact
#https://docs.google.com/spreadsheets/d/1sp6Ft_VUjt0L4ks6x77Z02Lt0TW8eBCj82MDnnfm01A/edit
#https://m.blog.naver.com/lch8422/221091626786 外
#https://forum.benchmark.rs/threads/ico-initial-coin-offering-diskusija.406332/page-2 外
#https://crypto500.news/yieldfinancefarm-next-generation-rewards-maximizing-defi-farm-2nd-yif-presale-is-live/ rewards
#https://cryptonewsbtc.org/2020/12/24/yieldfinancefarm-next-generation-rewards-maximizing-defi-farm-2nd-yif-presale-is-live/ 无
#https://bitco.news/yieldfinancefarm-next-generation-awards-making-the-most-of-defi-farm-2nd-presale-yif-live.html
#https://es.bitcoinethereumnews.com/technology/yieldfinancefarm-next-generation-rewards-maximizing-defi-farm-2nd-yif-presale-is-live/
#https://gitter.im/ethereum/welcome/archives/2018/03/19 社交媒体
#https://sysadmins.ru/topic535581-20.html 外
#https://forum.bitcoin.com/japanese/tenx-ico-50-t29406.html token sale
#https://fundaria.com/bounty/spreadsheet.php bounty
#https://opensea.io/0x29d7d1dd5b6f9c864d9db560d72a247c178ae86b?tab=activity nft opensea airdrop
#http://119.29.176.244:8001/position/tomatosfinance/7day
#https://www.coinlore.com/coin/tls-token
#https://textrunet.ru/raznoe/https-www-yandex-ru-koshelek-kak-sozdat-yandeks-koshelek-yandeks-kyu.html 外
#https://telegra.ph/Vitalik-Buterin-Founder-of-Ethereum-Soon-the-ether-will-pass-the-5000-mark-BTC-ETH-10-06 giveaway
#https://www.reddit.com/r/dutch/comments/jw4wn1/bitcoins_spam/ 外
#https://www.coinlore.com/coin/archetypal-network
#https://docs.google.com/spreadsheets/d/1w7KdVRPrCphMnTNWxc2EpKLCdQ6s3bd0UarSZ5fKLz8
#https://accapidocs.icomplyico.com/ 出现在api中
#https://telegra.ph/Vitalik-Buterin-Announces-100000-Ether-Giveaway-for-COVID-19-04-24 giveaway
#https://about.me/vitalik.buterin 返利
#https://tgstat.com/uz/channel/@airdrop_rex
#http://119.29.176.244:8001/position/kvaultfinance/7day 无
#https://textrunet.ru/raznoe/https-www-yandex-ru-koshelek-kak-sozdat-yandeks-koshelek-yandeks-kyu.html 重复
#https://www.facebook.com/kenhtintuctiendientu/photos/a.970253126364351/2693221900734123/?type=3 外
#https://medium.com/@horizon0x/horizon0x-ann-ddbb420af878 返利
#https://iconow.net/ico-1995/ ico 返利
#https://archived.moe/biz/thread/16821303 匿名论坛
#https://telegra.ph/Ethereum-Foundation-50000-ETH-Giveaway-Airdrop-12-01 giveaway
#http://panemicav2.rf.gd/ 触发metamask钓鱼检测
#https://teletype.in/@vitaliketh/q9Zw1wh-e giveaway airdrop
#https://tgstat.com/channel/@HyipCoins_airdrop/1302
#https://www.reddit.com/r/airdrops/comments/aucdbz/salvus_solutions_airdrop/ airdrop
#https://socialgrep.com/search?query=salvus airdrop
#https://i.warosu.org/biz/thread/22802914
#https://buybitcoin.lol/messages3.html
#https://t.me/s/EthereumPrivateChannel
#https://ne-np.facebook.com/hashtag/ether_private/
#https://eth-promo.wixsite.com/special-gift-event
#https://medium.com/@techteamlibracredit/welcome-to-libra-credit-community-d004b1731f9b
#https://tgstat.com/channel/@ClipClapsAnnouncements
#https://telemetr.io/en/channels/1245666043-clipclapsannouncements
#https://ne-np.facebook.com/groups/1873948109582411//
#https://www.haobi8.com/biwd/34041/ 科普
#http://ethereumforum.ru/threads/48/ 无
#https://nenettegarcia.com/jpr1u7v/who-is-the-publisher/smart-contract-address-ethereum.html 无
#https://telemetr.io/es/channels/1339444938-wbtc_network
#https://www.coinlore.com/coin/sense
#https://hi-in.facebook.com/coinalphax4r/posts/-new-born-metacom-token-on-ethereum-smart-chain-token-name-metacom-symbol-meta-s/184823243816795/ 外
#https://medium.com/@crypto_disrupt/rightbtc-release-their-official-api-87a5ee825995 科普
#https://www.reddit.com/user/bilogate/comments/ 报告
#https://zhuanlan.zhihu.com/p/36128198 报告
#https://www.facebook.com/mympv2/ 返利
#https://www.bleepingcomputer.com/news/security/beware-of-fake-john-mcafee-and-tesla-cryptocurrency-giveaways/ 报告
#https://www.hetianlab.com/html/news/news-2019062601.html 报
#https://nsanedown.superproxy.me/tags/tesla/ 报
#https://4hou.win/wordpress/?cat=3969 报



def twoaddr():
    addr1 = '0xc8b759860149542a98a3eb57c14aadf59d6d89b9'
    addr2 = '0x3b46c790ff408e987928169bd1904b6d71c00305'
    df1 = pandas.read_csv('ntx.csv')
    df2 = pandas.read_csv('itx.csv')
    timestamp1 = 1546272000
    timestamp2 = 1577808000
    time2num1 = {}
    time2num2 = {}
    for index,row in df1.iterrows():
        if (isinstance(row['to'],str) and row['to'] == addr1) or (isinstance(row['from'],str) and row['from'] == addr1) and row['timeStamp'] > timestamp1 and row['timeStamp'] < timestamp2:
            mytime = datetime.datetime.fromtimestamp(row['timeStamp'])
            mytime = mytime.strftime("%Y-%m-%d %H:%M")
            time2num1[mytime] = time2num1.get(mytime,0) + 1
        if (isinstance(row['to'],str) and row['to'] == addr2) or (isinstance(row['from'],str) and row['from'] == addr2) and row['timeStamp'] > timestamp1 and row['timeStamp'] < timestamp2:
            mytime = datetime.datetime.fromtimestamp(row['timeStamp'])
            mytime = mytime.strftime("%Y-%m-%d %H:%M")
            time2num2[mytime] = time2num2.get(mytime,0) + 1
    for index,row in df2.iterrows():
        if (isinstance(row['to'],str) and row['to'] == addr1) or (isinstance(row['from'],str) and row['from'] == addr1) and row['timeStamp'] > timestamp1 and row['timeStamp'] < timestamp2:
            mytime = datetime.datetime.fromtimestamp(row['timeStamp'])
            mytime = mytime.strftime("%Y-%m-%d %H:%M")
            time2num1[mytime] = time2num1.get(mytime,0) + 1
        if (isinstance(row['to'],str) and row['to'] == addr2) or (isinstance(row['from'],str) and row['from'] == addr2) and row['timeStamp'] > timestamp1 and row['timeStamp'] < timestamp2:
            mytime = datetime.datetime.fromtimestamp(row['timeStamp'])
            mytime = mytime.strftime("%Y-%m-%d %H:%M")
            time2num2[mytime] = time2num2.get(mytime,0) + 1
    with open('twoaddrTime2num1.txt','w') as f:#因为数据不全而统计错误
        print(time2num1,file=f)
    with open('twoaddrTime2num2.txt','w') as f:
        print(time2num2,file=f)
    with open('twoaddrTime2num1.txt','r') as f:
        twoaddrTime2num1 = literal_eval(f.read())
    with open('twoaddrTime2num2.txt','r') as f:
        twoaddrTime2num2 = literal_eval(f.read())
    x = twoaddrTime2num1.keys()
    y = twoaddrTime2num1.values()
    zipped = zip(x, y)
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))#根据时间排序
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    # print(result)
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c='0.1')
    x = twoaddrTime2num2.keys()
    y = twoaddrTime2num2.values()
    zipped = zip(x, y)
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))#根据时间排序
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    plt.scatter(x, y, c='0.9')
    plt.xticks(rotation=30)
    x_major_locator = MultipleLocator(10)
    ax = plt.gca()
    ax.set_ylabel('# of txs')
    ax.set_xlabel('Time')
    # plt.yscale('log')
    ax.xaxis.set_major_locator(x_major_locator)
    plt.savefig('fig5.jpg', bbox_inches='tight')
    plt.show()
def twoaddrcsv():
    addr1 = '0xc8b759860149542a98a3eb57c14aadf59d6d89b9'
    addr2 = '0x3b46c790ff408e987928169bd1904b6d71c00305'
    filename1 = addr1 + '_0.json'
    filename2 = addr2 + '_0.json'#不直接通过json统计两个地址的min2num（各分钟的交易数）
    with open('2addrtxs1.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['from', 'to', 'timeStamp'])
        with open(filename1,'r') as f:
            try:
                data = json.load(f)
                for txhash, mylist in data.items():
                    writer.writerow([mylist[0], mylist[1], mylist[3]])
            except Exception:
                traceback.print_exc()
    with open('2addrtxs2.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['from', 'to', 'timeStamp'])
        with open(filename2, 'r') as f:
            try:
                data = json.load(f)
                for txhash, mylist in data.items():
                    writer.writerow([mylist[0], mylist[1], mylist[3]])
            except Exception:
                traceback.print_exc()
def twocsv2fig5():
    addr1 = '0xc8b759860149542a98a3eb57c14aadf59d6d89b9'
    addr2 = '0x3b46c790ff408e987928169bd1904b6d71c00305'
    # df1 = pandas.read_csv('2addrtxs1.csv')
    # df2 = pandas.read_csv('2addrtxs2.csv')
    # time2num1 = {}
    # time2num2 = {}
    # for index,row in df1.iterrows():
    #     if (isinstance(row['to'], str) and row['to'] == addr1) or (isinstance(row['from'], str) and row['from'] == addr1) and row['timeStamp'] < '01-28':
    #         mytime = list(row['timeStamp'][5:-9])
    #         mytime[5] = ' '
    #         mytime = ''.join(mytime)
    #         time2num1[mytime] = time2num1.get(mytime, 0) + 1
    #     if (isinstance(row['to'], str) and row['to'] == addr2) or (isinstance(row['from'], str) and row['from'] == addr2):
    #         mytime = list(row['timeStamp'][5:-9])
    #         mytime[5] = ' '
    #         mytime = ''.join(mytime)
    #         time2num2[mytime] = time2num2.get(mytime, 0) + 1
    # for index,row in df2.iterrows():
    #     if (isinstance(row['to'], str) and row['to'] == addr1) or (isinstance(row['from'], str) and row['from'] == addr1) and row['timeStamp'] < '01-28':
    #         mytime = list(row['timeStamp'][5:-9])
    #         mytime[5] = ' '
    #         mytime = ''.join(mytime)
    #         time2num1[mytime] = time2num1.get(mytime, 0) + 1
    #     if (isinstance(row['to'], str) and row['to'] == addr2) or (isinstance(row['from'], str) and row['from'] == addr2):
    #         mytime = list(row['timeStamp'][5:-9])
    #         mytime[5] = ' '
    #         mytime = ''.join(mytime)
    #         time2num2[mytime] = time2num2.get(mytime, 0) + 1
    # with open('twoaddrTime2num1.txt','w') as f:
    #     print(time2num1,file=f)
    # with open('twoaddrTime2num2.txt','w') as f:
    #     print(time2num2,file=f)
    with open('twoaddrTime2num1.txt','r') as f:
        twoaddrTime2num1 = literal_eval(f.read())
    with open('twoaddrTime2num2.txt','r') as f:
        twoaddrTime2num2 = literal_eval(f.read())
    x = twoaddrTime2num1.keys()
    y = twoaddrTime2num1.values()
    zipped = zip(x, y)
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))#根据时间排序
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    # print(result)
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c='0.2', marker='o', label=addr1)
    x = twoaddrTime2num2.keys()
    y = twoaddrTime2num2.values()
    zipped = zip(x, y)
    sort_zipped = sorted(zipped, key=lambda x: (x[0]))#根据时间排序
    result = zip(*sort_zipped)
    x, y = [list(x) for x in result]
    plt.scatter(x, y, c='0.9', marker='o', label=addr2)
    plt.xticks(rotation=30)
    x_major_locator = MultipleLocator(150)
    ax = plt.gca()
    ax.set_ylabel('# of txs')
    ax.set_xlabel('Time')
    # plt.yscale('log')
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend()
    plt.savefig('fig5.jpg', bbox_inches='tight')
    plt.show()
def tgcrawler():
    admin_id = 12314567  # your chat id
    api_id = 10754897  # your telegram api id
    api_hash = 'e42a8b8fa4fc81078852b8ed3a14feb1'  # your telegram api hash
    bot_token = '5304906015:AAEAj2ihzk8Edkfm3J3z6bzGu4HmBWebrYM'  # your bot_token

async def main():
    api_id = 10754897  # your telegram api id
    api_hash = 'e42a8b8fa4fc81078852b8ed3a14feb1'  # your telegram api hash
    client = TelegramClient('session_name', api_id, api_hash)
    await client.start()

def gcnFig():
    precision = pd.read_csv('model_gcn_precision.csv')
    recall = pd.read_csv('model_gcn_recall.csv')
    f1 = pd.read_csv('model_gcn_f1.csv')
    x1 = precision['epoch']
    y1 = precision['value']
    x2 = recall['epoch']
    y2 = recall['value']
    x3 = f1['epoch']
    y3 = f1['value']
    plt.figure(figsize=(6, 6))
    plt.plot(x1, y1, label='precision', ls='--', c='black')
    plt.plot(x2, y2, label='recall', ls='-.', c='black')
    plt.plot(x3, y3, label='f1', ls='-', c='black')
    plt.xticks(rotation=30)
    ax = plt.gca()
    ax.set_ylabel('GCN')
    ax.set_xlabel('Epoch')
    plt.legend()
    plt.savefig('model_gcn.jpg', bbox_inches='tight')
    plt.show()
def gdcFig():
    precision = pd.read_csv('model_gdc_precision.csv')
    recall = pd.read_csv('model_gdc_recall.csv')
    f1 = pd.read_csv('model_gdc_f1.csv')
    x1 = precision['epoch']
    y1 = precision['value']
    x2 = recall['epoch']
    y2 = recall['value']
    x3 = f1['epoch']
    y3 = f1['value']
    plt.figure(figsize=(6, 6))
    plt.plot(x1, y1, label='precision', ls='--', c='black')
    plt.plot(x2, y2, label='recall', ls='-.', c='black')
    plt.plot(x3, y3, label='f1', ls='-', c='black')
    plt.xticks(rotation=30)
    ax = plt.gca()
    ax.set_ylabel('GDC')
    ax.set_xlabel('Epoch')
    plt.legend()
    plt.savefig('model_gdc.jpg', bbox_inches='tight')
    plt.show()
def tagcnFig():
    precision = pd.read_csv('model_tagcn_precision.csv')
    recall = pd.read_csv('model_tagcn_recall.csv')
    f1 = pd.read_csv('model_tagcn_f1.csv')
    x1 = precision['epoch']
    y1 = precision['value']
    x2 = recall['epoch']
    y2 = recall['value']
    x3 = f1['epoch']
    y3 = f1['value']
    plt.figure(figsize=(6, 6))
    plt.plot(x1, y1, label='precision', ls='--', c='black')
    plt.plot(x2, y2, label='recall', ls='-.', c='black')
    plt.plot(x3, y3, label='f1', ls='-', c='black')
    plt.xticks(rotation=30)
    ax = plt.gca()
    ax.set_ylabel('TAGCN')
    ax.set_xlabel('Epoch')
    plt.legend()
    plt.savefig('model_tagcn.jpg', bbox_inches='tight')
    plt.show()
def clustergcnFig():
    precision = pd.read_csv('model_cluster_gcn_precision.csv')
    recall = pd.read_csv('model_cluster_gcn_recall.csv')
    f1 = pd.read_csv('model_cluster_gcn_f1.csv')
    x1 = precision['epoch']
    y1 = precision['value']
    x2 = recall['epoch']
    y2 = recall['value']
    x3 = f1['epoch']
    y3 = f1['value']
    plt.figure(figsize=(6, 6))
    plt.plot(x1, y1, label='precision', ls='--', c='black')
    plt.plot(x2, y2, label='recall', ls='-.', c='black')
    plt.plot(x3, y3, label='f1', ls='-', c='black')
    plt.xticks(rotation=30)
    ax = plt.gca()
    ax.set_ylabel('CLUSTER-GCN')
    ax.set_xlabel('Epoch')
    plt.legend()
    plt.savefig('model_cluster-gcn.jpg', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # try:
    #     print("ntxs1")
    #     getNtxs1()
    # except Exception:
    #     pass
    # try:
    #     print("ntxs2")
    #     getnormaladdrNtxs2()
    # except Exception:
    #     pass
    # try:
    #     print("ntxs3")
    #     getnormaladdrNtxs3()
    # except Exception:
    #     pass
    # try:
    #     print("ntxs4")
    #     getnormaladdrNtxs4()
    # except Exception:
    #     pass
    # try:
    #     print("ntxs5")
    #     getnormaladdrNtxs5()
    # except Exception:
    #     pass
    # try:
    #     print("ntxs6")
    #     getnormaladdrNtxs6()
    # except Exception:
    #     pass
    # try:
    #     print("ntxs7")
    #     getnormaladdrNtxs7()
    # except Exception:
    #     pass
    # try:
    #     print("ntxs8")
    #     getnormaladdrNtxs8()
    # except Exception:
    #     pass
    # try:
    #     print("ntxs9")
    #     getnormaladdrNtxs9()
    # except Exception:
    #     pass
    # try:
    #     print("ntxs10")
    #     getnormaladdrNtxs10()
    # except Exception:
    #     pass
    # try:
    #     print("ntxs11")
    #     getnormaladdrNtxs11()
    # except Exception:
    #     pass
    # test()
    # try:
    #     print("ntxs12")
    #     getnormaladdrNtxs12()
    # except Exception:
    #     pass
    # try:
    #     print("ntxs13")
    #     getnormaladdrNtxs13()
    # except Exception:
    #     pass
    # try:
    #     print("itxs1")
    #     getnormaladdrItxs1()
    # except Exception:
    #     pass
    # try:
    #     print("itxs2")
    #     getnormaladdrItxs2()
    # except Exception:
    #     pass
    # try:
    #     print("itxs3")
    #     getnormaladdrItxs3()
    # except Exception:
    #     pass
    # try:
    #     print("itxs4")
    #     getnormaladdrItxs4()
    # except Exception:
    #     pass
    # try:
    #     print("itxs5")
    #     getnormaladdrItxs5()
    # except Exception:
    #     pass
    # try:
    #     print("itxs6")
    #     getnormaladdrItxs6()
    # except Exception:
    #     pass
    # try:
    #     print("itxs7")
    #     getnormaladdrItxs7()
    # except Exception:
    #     pass
    # try:
    #     print("itxs8")
    #     getnormaladdrItxs8()
    # except Exception:
    #     pass
    # try:
    #     print("itxs9")
    #     getnormaladdrItxs9()
    # except Exception:
    #     pass
    # try:
    #     print("itxs10")
    #     getnormaladdrItxs10()
    # except Exception:
    #     pass
    # try:
    #     print("etxs1")
    #     getnormaladdrEtxs1()
    # except Exception:
    #     pass
    # try:
    #     print("etxs2")
    #     getnormaladdrEtxs2()
    # except Exception:
    #     pass
    # try:
    #     print("etxs3")
    #     getnormaladdrEtxs3()
    # except Exception:
    #     pass
    # try:
    #     print("etxs4")
    #     getnormaladdrEtxs4()
    # except Exception:
    #     pass
    # try:
    #     print("etxs5")
    #     getnormaladdrEtxs5()
    # except Exception:
    #     pass
    # try:
    #     print("etxs6")
    #     getnormaladdrEtxs6()
    # except Exception:
    #     pass
    # try:
    #     print("etxs7")
    #     getnormaladdrEtxs7()
    # except Exception:
    #     pass
    # try:
    #     print("etxs8")
    #     getnormaladdrEtxs8()
    # except Exception:
    #     pass
    # try:
    #     print("etxs")
    #     getnormaladdrEtxs9()
    # except Exception:
    #     pass
    # pass
    # try:
    #     print("itxs1")
    #     getItxs1()
    # except Exception:
    #     print("itxs1")
    # try:
    #     print("itxs2")
    #     getItxs2()
    # except Exception:
    #     print("itxs2")
    # try:
    #     print("itxs3")
    #     getItxs3()
    # except Exception:
    #     print("itxs3")
    # try:
    #     print("itxs4")
    #     getItxs4()
    # except Exception:
    #     print("itxs4")
    # try:
    #     getEtxs1()
    # except Exception:
    #     pass
    # try:
    #     print("etxs2")
    # normalLive2()
    # norAddrA3OutItxTx()
    # norAddr3IOTxnum()
    # try:
    #     print("norAddrA3InNtxTime")
    #     norAddrA3InNtxTime()
    # except Exception:
    #     traceback.print_exc()
    # try:
    #     print("norAddrA3OutNtxTime")
    #     norAddrA3OutNtxTime()
    # except Exception:
    # #     traceback.print_exc()
    # try:
    #     print("norAddrA3Intxnum")
    #     norAddrA3Intxnum()
    # except Exception:
    #     traceback.print_exc()
    # try:
    #     print("norAddrA3Outtxnum")
    #     norAddrA3Outtxnum()
    # except Exception:
    #     traceback.print_exc()
    # mixfig9Livingtime()
    # fig43outtxnum()
    # scamProfitRank()
    # fig14()
    # fig12x()
    # tab1n()
    # tab1i()
    # tab1e()
    # fig15b()
    # mixfig9livcon()
    # dprmatrix()
    # fig4()
    # fig7n()
    # util()
    # fig12expansion()
    # unknownNodeNtx()
    # unknownNodeNtx()
    # es2csv()
    # txsimplify()
    # fig16aData()
    # scamVictim()
    # gephifig16b()
    # beforeGroupProfit()
    # scamUnknownVictim()
    # fig12expansion2FromTo()
    # es2csv()
    # fig16aStats()
    # gephifig16b()
    # myimg2pdf()
    # unknownVictim()
    # gephifig16a()
    # addr2expgroup()
    # scamUnknownVictim()
    # util()
    # fig4()
    # mixfig9Livingtime()
    # fig7iCdf()
    # fig7nCdf()
    # myimg2pdf()
    fig43intxnum()
    fig43outtxnum()
    myimg2pdf()
    # tweetcsv()
    # website()
    # gcnFig()
    # gdcFig()
    # tagcnFig()
    # clustergcnFig()
    # asyncio.run(main())
    # testSele()
    # alltg()
    # rightfig14()
    # twoaddr()
    # twoaddrcsv()
    # twocsv2fig5()
    # mixfig6()
    # fig15a()
    # tweet()
    # fig2()
    # mixfig3()
    # normalAddrtx2csv1()
    # normalAddrtx2csv2()
    # getComponentNode()
    # getCompNodeCsv()
    # attackedEdge()
    # fig7n()
    # dprmatrixc()
    # getnpz()
    # fig16b()
    # fig15a()
    # scamAvgIncomeOutcome()
    # norAddrA3LivingTime()
#gcn gdc tagcn
#收集整理大量数据时，尽量保存中间文件，即使由于机器性能原因或者ide设置原因等中断运行，也能避免效率的降低。
#涉及网络爬虫的工作中可能会出现由于当时的网络原因出现问题，包括但不限于整个代码停止运行

#找和欺诈地址有关的推特，以及具体的欺诈地址举例
# 以及欺诈报告相关报告和地址