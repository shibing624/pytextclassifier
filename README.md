![alt text](docs/logo.svg)


[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/deepmipt/DeepPavlov/blob/master/LICENSE) ![](https://img.shields.io/badge/Language-Python-blue.svg) ![](https://img.shields.io/badge/Python-3.X-red.svg)

# text-classifier
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
### Requirements and Installation
```
git clone https://github.com/shibing624/text-classifier.git
pip3 install -r requirements.txt
```

### Example Usage
```
cd text-classifier
vim config.py
```

1. Preprocess with segment
```
python3 preprocess.py
```

2. Train model

you can change model with edit `config.py` and train model.
```
python3 train.py
```

3. Predict with test data
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
