# text-classifier
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/deepmipt/DeepPavlov/blob/master/LICENSE) ![](https://img.shields.io/badge/Language-Python-blue.svg) ![](https://img.shields.io/badge/Python-3.X-red.svg)


Text classifier and cluster. It can be applied to the fields of sentiment polarity analysis, text risk classification and so on, and it supports multiple classification algorithms.

-----


**text-classifier** s a python Open Source Toolkit for Chinese text categorization. The goal is to implement text categorization algorithm, so as to achieve the use in the generative environment. **text-classifier** has the characteristics of clear algorithm, high performance and customizable corpus.

**text-classifier** provides the following functions：
> * Classifier
  * LogisticRegression
  * MultinomialNB
  * KNN
  * SVM
  * RandomForest
  * DecisionTreeClassifier
  * Xgboost
  * Neural Network
> * Evaluate
  * Precision
  * Recall
  * F1
> * Test
  * Chi-square test
> * Cluster
  * MiniBatchKmeans

While providing rich functions, **text-classifier** internal modules adhere to low coupling, model adherence to inert loading, dictionary publication, and easy to use.

------
## demo 

https://www.borntowin.cn/product/sentiment_classify

------

## Usage
1. 获取代码与安装依赖：
```
git clone https://github.com/shibing624/text-classifier.git
pip3 install -r requirements.txt
```

2. 修改配置文件：
```
cd text-classifier
vim config.py
```

3. 预处理文本文件（文本切词）：
```
python3 proprecess.py
```

4. 训练模型
```
python3 train.py
```

5. 模型测试或预测
```
python3 infer.py
```


## Algorithm
  - [x] LogisticRegression
  - [x] Random Forest
  - [x] Decision Tree
  - [x] K-Nearest Neighbours
  - [x] Naive bayes
  - [x] Xgboost
  - [x] Support Vector Machine(SVM)
  - [x] MLP
  - [x] Ensemble
  - [x] Stack
  - [x] Xgboost_lr
  - [x] text CNN
  - [x] text RNN
  - [x] fasttext
  - [x] HAN
  - [x] Kmenas


## Thanks
  - SentimentPolarityAnalysis

## Licence
  - Apache Licence 2.0
