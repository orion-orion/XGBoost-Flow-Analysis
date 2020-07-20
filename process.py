import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from html.parser import HTMLParser
import re
from urllib import parse

def DecodeQuery(fileName):
    data = [x.strip() for x in open(fileName, "r").readlines()]
    query_list = []
    for item in data:
        item = item.lower()
        if len(item) > 50 or len(item) < 5:
             continue        
        h = HTMLParser()
        item = h.unescape(item)
        item = parse.unquote(item)
        item, number = re.subn(r'\d+', "8", item)
        item, number = re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?:]+', "http://u", item)
        query_list.append(item)
    return list(set(query_list))   

def readFile():
    #读取训练集数据
    vectorizer =TfidfVectorizer(ngram_range=(1,3))
    bX1_d = DecodeQuery('./data/恶意软件流量url.txt')
    bX2_d = DecodeQuery('./data/网络攻击流量url.txt')
    gX_d = DecodeQuery('./data/goodx.txt')
    X_train=pd.DataFrame(vectorizer.fit_transform(bX1_d+bX2_d+gX_d).todense())
    Y_train=[0]*len(gX_d)+[1]*len(bX1_d)+[2]*len(bX2_d) #正常请求标签为0 恶意软件流量标签为1 网络攻击流量标签为2
    #读取测试集数据
    X_test_d=DecodeQuery('./data/testx.txt')
    X_test =pd.DataFrame(vectorizer.transform(X_test_d).todense())
    return pd.DataFrame(X_train),pd.DataFrame(Y_train),pd.DataFrame(X_test)
  

