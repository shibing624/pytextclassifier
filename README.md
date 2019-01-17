# text-classifier
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/deepmipt/DeepPavlov/blob/master/LICENSE) ![](https://img.shields.io/badge/Language-Python-blue.svg) ![](https://img.shields.io/badge/Python-3.X-red.svg)


Text classifier. It can be applied to the fields of sentiment polarity analysis, text risk classification and so on, and it supports multiple classification algorithms.

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

While providing rich functions, **text-classifier** internal modules adhere to low coupling, model adherence to inert loading, dictionary publication, and easy to use.

------
## demo 

http://www.borntowin.cn/product/sentiment_classify

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
  - [done] LogisticRegression
  - [done] Random Forest
  - [done] Decision Tree
  - [done] K-Nearest Neighbours
  - [done] Naive bayes
  - [done] Xgboost
  - [done] Support Vector Machine(SVM)
  - [done] MLP
  - [done] Ensemble
  - [done] Stack
  - [done] Xgboost_lr
  - [done] text CNN
  - [done] text RNN
  - [done] fasttext
  - [done] HAN


## Thanks
  - SentimentPolarityAnalysis

## Licence
  - Apache Licence 2.0
