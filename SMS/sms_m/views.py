from django.shortcuts import render
from django import forms
import sys, os
from time import time
import json, pickle
from scipy import sparse, io
from numpy import *
import jieba
import jieba.posseg as pseg     # posseg是为了分词后，带有词性
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.linear_model import Perceptron


# Create your views here.
class GetSms(forms.Form):
    longstr = forms.CharField()
    dealwith = forms.CharField()
    algo = forms.CharField()


def index(request):
    result_str = '请将短信内容填入上面表单！！'
    if request.method == 'GET':
        form = GetSms()
        return render(request, 'index.html', {'result_str': result_str})
    elif request.method == 'POST':
        form = GetSms(request.POST)
        print(form)
        if form.is_valid():
            long_str = form.cleaned_data['longstr']             # 短信内容
            result_str2 = long_str                              # 返回短信内容
            deal_with = form.cleaned_data['dealwith']           # 数据处理方式
            algo = form.cleaned_data['algo']                    # 使用的算法  # 这三个都是字符串类型
            # 这里之间写处理代码
            # 文本获取及处理
            x_word = []
            text = long_str.strip()
            words = pseg.cut(text)
            new_text = ''.join(w.word for w in words if w.flag != 'x')
            words = jieba.cut(new_text)
            x_word.append(' '.join(i for i in words))
            with open(os.path.dirname(os.path.dirname(__file__)) + '/model/tfidf', 'rb') as fp:
                vec_tfidf = pickle.load(fp)
            test_x = vec_tfidf.transform(x_word)
            test_x = test_x.todense()
            if deal_with == 'pca':
                with open(os.path.dirname(os.path.dirname(__file__)) + '/model/pca', 'rb') as fp:
                    pca = pickle.load(fp)
                test_x = pca.transform(test_x)
            elif deal_with == 'nmf':
                with open(os.path.dirname(os.path.dirname(__file__)) + '/model/nmf', 'rb') as fp:
                    nmf = pickle.load(fp)
                test_x = nmf.transform(test_x)

            with open(os.path.dirname(os.path.dirname(__file__)) + '/model/' + deal_with + 'Model/' + algo, 'rb') as fp:
                model = pickle.load(fp)
            y_lable = model.predict(test_x)

            ####
            if y_lable == 1:
                ans = '垃圾短信'
            else:
                ans = '正常短信'
            result_str = '经' + deal_with + '+' + algo + '算法初步判定该短信为： ' + ans       # 数据返回接口
            # print(result_str)
    return render(request, 'index.html', {'result_str': result_str, 'result_str2': result_str2})
