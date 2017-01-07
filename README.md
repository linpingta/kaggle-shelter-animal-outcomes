# Kaggle比赛 [shelter-animal-outcome](https://www.kaggle.com/c/shelter-animal-outcomes/leaderboard)

这是我参加shelter-animal-outcome比赛的项目

我在比赛中的排名是 79th/1604。

运行脚本需要满足以下python模块被安装：

    numpy / pandas / scikit-learn / xgboost

需要将train.csv、test.csv和submission.csv置于data/内，
或在conf/simple.conf中替换以下目录为实际train.csv、test.csv和submission.csv目录：

    train_filename=./data/train.csv
    test_filename=./data/test.csv
    submission_filename=./data/submission.csv

然后运行 bin目录下的 shelter.py

    python bin/shelter.py
  
在conf/simple.conf中的以下配置控制操作：

    do_train=1          #是否训练
    do_validation=1     #是否做validation
    do_search_parameter #是否做参数搜索 （需要在代码中对应指定参数范围）
    do_test=1           #是否测试

项目文件结构：

    bin /
      simple.py / shelter.py : 与最终版本的项目结果相关
    lib /
      * ：与最终版本的项目结果相关
  
    bin.bk /
      *_simple.py , model_average.py : 历史尝试，包括单独使用的随机森林，KNN, XGBoost以及model融合尝试

关于项目的介绍请见我的[博客](https://linpingta.github.io/blog/2016/07/10/kaggle-shelter-animal-outcome/)。

我在项目的bin.bk中保留了所有历史版本，并且没有对其中的格式做修正，主要原因在于我在初期并没有一个合适的model training模板用于尝试不同的模型方法，
后续在各种模型方法上统一后，我开发了我的[model trainer模板方法](https://github.com/linpingta/tools/tree/master/model_trainer)，并在
最终版本的提交中使用了相应的结构。
  
