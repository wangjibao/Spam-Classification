#  -*- coding: utf-8 -*-

from time import time
import json
from scipy import sparse, io
from numpy import *
import jieba
import pickle
import jieba.posseg as pseg     # posseg是为了分词后，带有词性
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.externals import joblib


def process_compress(n_cases, dimensionality_reduction_type, n_components):

    # 01读取文件，将短信标签保存在 y_lable 中，短信的文本部分保存在 x_text 中
    y_lable = []
    x_text = []
    negatives = 0
    positives = 0
    with open('data/带标签短信.txt', 'r', encoding='utf-8') as fp:
        for line in fp.readlines():   # 正负样本参半
            if line[0] == '1' and positives < n_cases/2:
                positives += 1
            elif line[0] == '0' and negatives < n_cases/2:
                negatives += 1
            else:
                continue
            y_lable.append(int(line[0]))
            x_text.append(line[1:].strip())
    with open('data/lable', 'w') as fp:
        json.dump(y_lable, fp)

    # 02通过 jieba 分词对 x_text 进行处理，如词语间空格分开(后期生成 tf-idf 矩阵就是凭借空格识别词语的)，去除非语素词等
    x_word = []
    for text in x_text:
        words = pseg.cut(text)
        new_text = ''.join(w.word for w in words if w.flag != 'x')
        words = jieba.cut(new_text)
        x_word.append(' '.join(i for i in words))

    # 03构建词频矩阵，将文本中的词语转换成 tf-idf 矩阵
    vec_tfidf = TfidfVectorizer()  # Equivalent to CountVectorizer followed by TfidfTransformer.
    vec_tfidf.fit(x_word)
    with open('model/tfidf', 'wb') as fp:
        pickle.dump(vec_tfidf, fp)
    tfidf = vec_tfidf.transform(x_word)
    # print(vec_tfidf.vocabulary_)  # 词典，tfidf会自动去掉那些“的”，“嗯”等的不重要的词
    # print(vec_tfidf.get_feature_names())

    # 04对tfidf矩阵进行特征压缩提取并进行保存
    tfidf = tfidf.todense()
    with open('data/'+dimensionality_reduction_type, 'wb') as fp:
        if dimensionality_reduction_type == 'raw':
            io.mmwrite(fp, sparse.csr_matrix(tfidf))  # 矩阵的压缩存储
        elif dimensionality_reduction_type == 'pca':
            pca = PCA(n_components=n_components)     # 生成模型
            pca.fit(tfidf)                           # fit数据进行训练
            transform_tfidf = pca.transform(tfidf)    # 矩阵压缩存储
            io.mmwrite(fp,  transform_tfidf)
            with open('model/pca', 'wb') as f:
                pickle.dump(pca, f)
        elif dimensionality_reduction_type == 'nmf':
            nmf = NMF(n_components=n_components)
            nmf.fit(tfidf)
            transform_tfidf = nmf.transform(tfidf)
            io.mmwrite(fp, transform_tfidf)
            with open('model/nmf', 'wb') as f:
                pickle.dump(nmf, f)


if '__main__' == __name__:
    # if len(sys.argv) != 4 or sys.argv[1] not in ['raw', 'pca', 'nmf'] or int(sys.argv[2]) < 10:
    #     print('Usage Error!')
    #     print('python dataProcess.py n_cases dimensionality_reduction_type n_components')
    #     sys.exit()

    print('Data Process Start...')
    t0 = time()
    # n_cases = int(sys.argv[1])
    # dimensionality_reduction_type = sys.argv[2]
    # n_components = int(sys.argv[3])
    n_cases = 15000
    n_components = 1000

    dimensionality_reduction_type = 'raw'
    process_compress(n_cases, dimensionality_reduction_type, n_components)
    print('------------------------')

    dimensionality_reduction_type = 'pca'
    process_compress(n_cases, dimensionality_reduction_type, n_components)
    print('------------------------')

    dimensionality_reduction_type = 'nmf'
    process_compress(n_cases, dimensionality_reduction_type, n_components)

    print('Data Process End...')
    print('Data Processing Done in %.3fs...' % (time()-t0))


