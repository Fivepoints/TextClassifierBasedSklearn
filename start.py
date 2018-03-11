#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
 @desc:  
 @author: Zhao Pengya  
 @created: 2018/2/4 14:28  
 @software: PyCharm python 3.5.4
 
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold

from sklearn.externals import joblib
import numpy as np

# def preProcessing():
#     fw_data = open('data2.txt', 'a', encoding='utf-8')
#     fw_target = open('target2.txt', 'a', encoding='utf-8')
#     with open('labelWords2.txt', 'r', encoding='utf-8') as fr:
#         lines = fr.readlines()
#         for line in lines:
#             spts = line.strip().split(':')
#             label = spts[0]
#             data = spts[1].split('\t')
#             fw_target.write(label+'\n')
#             fw_data.write('\t'.join(data) + '\n')
#     fw_data.close()
#     fw_target.close()
#
# preProcessing()

def loadCorpus():
    data = []; label = []
    with open('data.txt', 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            data.append(' '.join(line.strip('\n').split('\t')))
    with open('target.txt', 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            label.append(int(line.strip()))
    print('load corpus is done!')
    return data, label


if __name__ == '__main__':
    # preProcessing()
    # 加载数据, data为分词后结果, label为类别标签
    data, label = loadCorpus()

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()

    # 得到文档向量矩阵
    X = vectorizer.fit_transform(data)
    tfidf = transformer.fit_transform(X)

    # 使用卡方检验进行特征提取
    model = SelectKBest(chi2, k=10000)
    chi = model.fit_transform(X, label)
    # 得到选择特征索引
    feature_index = model.get_support(indices=True)

    # 降维
    tfidf = tfidf[:, feature_index]

    word = vectorizer.get_feature_names()
    # 特征名称
    feature_names = [word[index] for index in feature_index]
    feature_names_other = list(set(word).difference(set(feature_names)))

    x = tfidf
    y = np.array(label)
    # 交叉验证
    skf = StratifiedKFold(n_splits=10)
    y_pre = y.copy()
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = MultinomialNB().fit(x_train, y_train)
        # 保存模型
        joblib.dump(clf, 'bayes_model_content.m')
        y_pre[test_index] = clf.predict(x_test)
    print('贝叶斯:准确率为 %.6f' % (np.mean(y_pre == y)))