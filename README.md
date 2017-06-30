# classifier-in-action
在情感分析领域中应用分类算法处理。python开发。

---


**classifier-in-action**是在情感分析领域中应用分类算法处理的python开源工具包，目标是提高情感极性分析的准确率，以达到在生成环境中使用。**classifier-in-action**具备算法清晰、性能高效、语料可自定义的特点。

**classifier-in-action**提供下列功能：
> * 分类器
  * 朴素贝叶斯(NB)
  * K最近邻(KNN)
  * 支持向量机(SVM)
  * 最大熵(MaxEnt)
  * 情感词典
> * 评估
  * 准确率
  * 召回率
  * F值
  * 训练、测试时间
  * 结果输出
> * 统计检验
  * 卡方检验(Chi-square test)



在提供丰富功能的同时，**classifier-in-action**内部模块坚持低耦合、模型坚持惰性加载、词典明文发布，使用方便。

------

## 调用方法


所有Demo都位于[demo](https://github.com/shibing624/classifier-in-action/demo)下，比文档覆盖了更多细节，强烈建议运行一遍。

#### 如何使用

  - Demo


	```
	from classifier.dict import DictClassifier
    result = DictClassifier().analyse_sentence("土豆丝我觉得很好吃",print_show=True)
    print(result)
	```


#### 特性
   - 支持结巴中文分词
   - 情感极性词典
   - 多种分类器
   - 多行业熟语料数据

#### 算法
  - [ ] K-Nearest Neighbours
  - [ ] Naive bayes
  - [ ] Maximum Entropy
  - [ ] Support Vector Machine
  - [ ] Dict

#### 性能评估
  - 效果比较
	
	```
	词典(Dict):
        准确率：准确率较高（80%以上），随着人工工作量的增加，准确率增加
        优点：易于理解
        缺点：人工工作量大
    
    kNN:
        准确率：很低（60% - 70%）
        优点：思想简单、算法简单
        缺点：准确率低；耗内存；耗时间
    
    Bayes:
        准确率：还可以（70% - 80%）
        优点：简单，高效，运算速度快，扩展性好
        缺点：准确率不高，达不到实用
    
    最大熵:
        准确率：比较高（83%以上）
        优点：准确率高
        缺点：训练时间久
    
    SVM:
        准确率：最高（85%以上）
        优点：准确率高
        缺点：训练耗时
	
	```

## 鸣谢
  - SentimentPolarityAnalysis 项目 

## 许可证
  许可证为Apache Licence 2.0