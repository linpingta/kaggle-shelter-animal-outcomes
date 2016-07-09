# Kaggle比赛 [shelter-animal-outcome](https://www.kaggle.com/c/shelter-animal-outcomes/leaderboard)

这是我参加shelter-animal-outcome比赛的项目，其中包含我用到过的所有文件和尝试（因此bin文件里看到起来非常dirty，下面我会解释）。

我在比赛中的排名是 54th/1229 （截止文档发布时2016.7.10，我会在比赛结束后更新这一数据）。

运行脚本需要满足以下python模块被安装：

    numpy / pandas / scikit-learn / xgboost

同时替换conf/shelter.conf中以下目录为实际train.csv、test.csv和submission.csv目录：

    train_filename=/home/test/model_trainer/data/train.csv
    test_filename=/home/test/model_trainer/data/test.csv
    submission_filename=/home/test/model_trainer/data/submission.csv

然后运行 bin目录下的 shelter.py

    python bin/shelter.py
  
在conf/shelter.conf中的以下配置控制操作：

    do_train=1          #是否训练
    do_validation=1     #是否做validation
    do_search_parameter #是否做参数搜索 （需要在代码中对应指定参数范围）
    do_test=1           #是否测试

项目文件结构：

    bin /
      simple.py / shelter.py : 与最终版本的项目结果相关
    lib /
      * ：与最终版本的项目结果相关
  
    bin /
      *_simple.py , model_average.py : 历史尝试，包括单独使用的随机森林，KNN, XGBoost以及model融合尝试

关于项目的介绍请见我的[博客](linpingta.cn)。

我在项目中保留了所有历史版本，并且没有对其中的格式做修正，主要原因在于我在初期并没有一个合适的model training模板用于尝试不同的模型方法，
后续在各种模型方法上统一后，我开发了我的[model trainer模板方法](https://github.com/linpingta/tools/tree/master/model_trainer)，并在
最终版本的提交中使用了相应的结构。因此项目目录显得比较乱，但我认为保留所有的尝试（我最终的选择是XGBoost+feature engineering）对我以后
回忆项目中的尝试是一件好事（毕竟看起来这个项目是一个kaggle非rank point比赛，而且最后又出现了一些作弊策略的问题（我在博客中会详细描述））。
  
