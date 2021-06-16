![alt text](docs/logo.svg)


[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](./LICENSE) ![](https://img.shields.io/badge/Language-Python-blue.svg) ![](https://img.shields.io/badge/Python-3.X-red.svg) ![](https://img.shields.io/badge/Python-2.X-red.svg)

# text-classifier
Text classifier and cluster. It can be applied to the fields of sentiment polarity analysis, text risk classification and so on, and it supports multiple classification algorithms.

-----


**text-classifier** is a python Open Source Toolkit for text classification and text clustering. The goal is to implement text analysis algorithm, so as to achieve the use in the production environment. **text-classifier** has the characteristics of clear algorithm, high performance and customizable corpus.

**text-classifier** provides the following functions：
### Classifier
  - [x] LogisticRegression
  - [x] Random Forest
  - [x] Decision Tree
  - [x] K-Nearest Neighbours
  - [x] Naive bayes
  - [x] Xgboost
  - [x] Support Vector Machine(SVM)
  - [x] Xgboost
  - [x] Xgboost_lr
  - [x] MLP
  - [x] Ensemble
  - [x] Stack
  - [x] TextCNN
  - [x] TextRNN
  - [x] Fasttext

### Evaluate
  - [x] Precision
  - [x] Recall
  - [x] F1

  
### Test
  - [x] Chi-square test
  
### Cluster
  - [x] MiniBatchKmeans

While providing rich functions, **text-classifier** internal modules adhere to low coupling, model adherence to inert loading, dictionary publication, and easy to use.

------

## Usage
### Requirements and Installation
```
git clone https://github.com/shibing624/text-classifier.git
pip3 install -r requirements.txt
```

### Example Usage

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


## Thanks
  - SentimentPolarityAnalysis

## Licence
  - Apache Licence 2.0
