'''
这一节用来记录sklearn的pipeline的学习
'''

from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm import SVC,LinearSVC,LinearSVR
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


'''
pipeline类
共有3个参数
1. steps，是最重要的参数，主要用来设定流水线上的一道道工序，从左到右是流水线上的先后顺序，工序的类型为[(),()]，元祖里面的信息分别为名字和工序。
前面n-1道工序里面的方法必须要有fit，transform两个函数。最后一个工序为estimator，只需要有fit函数就行

2. memory：缓存地址

3. verbose，默认为False，用来显示每个流水线所消耗的时间
'''


selected_categories = [
    'comp.graphics',
    'rec.motorcycles',
    'rec.sport.baseball',
    'misc.forsale',
    'sci.electronics',
    'sci.med',
    'talk.politics.guns',
    'talk.religion.misc']

newsgroups_train=fetch_20newsgroups(subset='train',
                                    categories=selected_categories,
                                    remove=('headers','footers','quotes'))
newsgroups_test=fetch_20newsgroups(subset='train',
                                    categories=selected_categories,
                                    remove=('headers','footers','quotes'))

train_texts=newsgroups_train['data']
train_labels=newsgroups_train['target']
test_texts=newsgroups_test['data']
test_labels=newsgroups_test['target']


## 贝叶斯方法来对文本进行分类
## max_feature用来控制提取出来的向量维度最多为1000维，词的选择是根据词频来的
text_clf = Pipeline([('tfidf', TfidfVectorizer(max_features=1000)),('clf',(MultinomialNB()))])
text_clf.fit(train_texts, train_labels)
predicted =  text_clf.predict(test_texts)
print(np.mean(predicted==test_labels))
