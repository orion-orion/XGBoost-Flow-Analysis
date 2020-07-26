import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from html.parser import HTMLParser
import re
from urllib import parse
import numpy as np
import pymysql
import os
def DecodeQuery(fileName):
    data = [x.strip() for x in open(fileName, "r").readlines()]
    query_list = []
    for item in data:
        item = item.lower()
        #if len(item) > 50 or len(item) < 5:
         #    continue        
        h = HTMLParser()
        item = h.unescape(item) #将&gt或者&nbsp这种转义字符转回去
        item = parse.unquote(item)#解码,就是把字符串转成gbk编码，然后把\x替换成%。如果
        item, number = re.subn(r'\d+', "8", item) #正则表达式替换
        item, number = re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?:]+', "http://u", item)
        query_list.append(item)
    return list(set(query_list))   

def readFile(db):
    #读取训练集数据
    vectorizer =TfidfVectorizer(ngram_range=(1,3))
    bX1_d = DecodeQuery('./data/网络攻击.txt')
    bX2_d = DecodeQuery('./data/恶意软件.txt')
    gX_d = DecodeQuery('./data/业务流量.txt')
    X_train=vectorizer.fit_transform(bX1_d+bX2_d+gX_d).todense()
    Y_train=np.array([0]*len(gX_d)+[1]*len(bX1_d)+[2]*len(bX2_d)).reshape(-1,1) #正常请求标签为0  网络攻击流量标签为1 恶意软件流量标签为2
    train=np.concatenate((X_train,Y_train),axis=1)
    np.random.shuffle(train)
    X_train=train[:,:-1]
    #读取测试集数据
    os.system("rm /var/lib/mysql-files/testx.csv")
    cursor=db.cursor()
    cursor.execute('use EP2')
    cursor.execute(r'''SELECT * FROM url INTO OUTFILE '/var/lib/mysql-files/testx.csv' FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\r\n' ''')
    X_test_d=DecodeQuery('/var/lib/mysql-files/testx.csv')
    X_test =vectorizer.transform(X_test_d).todense()
    return pd.DataFrame(X_train),Y_train.ravel(),pd.DataFrame(X_test)
  
