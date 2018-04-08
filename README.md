# text-classifier
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/deepmipt/DeepPavlov/blob/master/LICENSE) ![](https://img.shields.io/badge/Language-Python-blue.svg) ![](https://img.shields.io/badge/Python-3.X-red.svg)


Text classifier. It can be applied to the fields of sentiment polarity analysis, text risk classification and so on, and it supports multiple classification algorithms.

-----


**text-classifier**s a python Open Source Toolkit for Chinese text categorization. The goal is to implement text categorization algorithm, so as to achieve the use in the generative environment.**text-classifier**has the characteristics of clear algorithm, high performance and customizable corpus.

**text-classifier**provides the following functions：
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

While providing rich functions, **text-classifier**internal modules adhere to low coupling, model adherence to inert loading, dictionary publication, and easy to use.

------

## Usage
* classic text classifier:
run [train.py](https://github.com/shibing624/text-classifier/blob/master/text-classifier/classic/train.py): 
```
cd classic
python train.py
```


* neural network text classifier:
run [train.py](https://github.com/shibing624/text-classifier/blob/master/text-classifier/neural_network/train.py): 
```
cd neural_network
python segment.py
python train_w2v_model.py
python train.py
```
    

## Algorithm
  - [done] LogisticRegression
  - [done] K-Nearest Neighbours
  - [done] Naive bayes
  - [done] Support Vector Machine
  - [done] Random Forest
  - [done] Decision Tree
  - [done] Xgboost
  - [done] [Neural Network](https://github.com/shibing624/text-classifier/tree/master/text-classifier/neural_network)


## Thanks
  - SentimentPolarityAnalysis

## Licence
  - Apache Licence 2.0