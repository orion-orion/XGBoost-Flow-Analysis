import numpy as np
import joblib
from numpy.lib.function_base import vectorize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from html.parser import HTMLParser
import re
from urllib import parse
def DecodeQuery(fileName):
    data = [x.strip() for x in open(fileName, "r").readlines()]
    query_list = []
    for item in data:
        item = item.lower()
        # if len(item) > 50 or len(item) < 5:
        #     continue        
        h = HTMLParser()
        item = h.unescape(item)
        item = parse.unquote(item)
        item, number = re.subn(r'\d+', "8", item)
        item, number = re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?:]+', "http://u", item)
        query_list.append(item)
    return list(set(query_list))
class Tfid():
    def __init__(self) -> None:
        self._vectorizer =TfidfVectorizer(ngram_range=(1,3))
class Classifier():
    def __init__(self):
        self._vectorizer =TfidfVectorizer(ngram_range=(1,3))
        self._NBM = [
            MultinomialNB(alpha=0.01),  # 多项式模型-朴素贝叶斯
            BernoulliNB(alpha=0.01),
            DecisionTreeClassifier(max_depth=100),
            RandomForestClassifier(criterion='gini', max_depth=100,n_estimators=200),
            LogisticRegression(random_state=40,solver='lbfgs', max_iter=10000, penalty='l2',multi_class='multinomial',class_weight='balanced', C=100),
            LinearSVC(class_weight='balanced',random_state=100, penalty='l2',loss='squared_hinge', C=0.92, dual=False),
            SVC(kernel='rbf', gamma=0.7, C=1),
            # GradientBoostingClassifier(param)
        ]
        self._NAME = ["多项式", "伯努利", "决策树", "随机森林", "线性回归", "linerSVC", "svc-rbf"]
    def fit(self,X,Y):
        #交叉验证
        x_train, x_test, y_train, y_test =train_test_split(X, Y, test_size=0.2, random_state=666)
        max_dts=-1
        for model, modelName in zip(self._NBM, self._NAME):
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            dts = len(np.where(pred == y_test)[0])/ len(y_test)
            print("{} 精度:{:.5f} ".format(modelName, dts * 100))
            if dts > max_dts:  #保存最佳模型s
                joblib.dump(model, 'model/model.pkl')
    def predict(self,x):
        clf = joblib.load('model/model.pkl')
        return clf.predict(x)
if __name__ =='__main__':
    #读取训练集数据
    tfid=Tfid()
    bX = DecodeQuery('./data/badx.txt')
    gX = DecodeQuery('./data/goodx.txt')
    gY = [1]*len(gX)   #正常标签为1
    bY = [0]*len(bX)   #恶意请求标签为0 bX = self.DecodeQuery('./data/badx.txt')
    X=tfid._vectorizer.fit_transform(bX+gX)
    Y=bY+gY

    #模型训练
    clf=Classifier()
    clf.fit(X,Y)

    #加载测试集
    X_test=DecodeQuery('./data/testx.txt')
    X_test2 = tfid._vectorizer.transform(X_test)

    #进行推断
    Y_hat=clf.predict(X_test2)
    res_list = []

    #打印结果
    for url , y in zip(X_test, Y_hat):
        label = '正常请求' if y == 1 else '恶意请求'
        print(label , url )


