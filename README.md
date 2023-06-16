# Kaggle Competition [shelter-animal-outcome](https://www.kaggle.com/c/shelter-animal-outcomes/leaderboard)

This is my project for the shelter-animal-outcome competition on Kaggle.

My rank in the competition is 79th / 1604 (https://www.kaggle.com/competitions/shelter-animal-outcomes/leaderboard)

To run the script, the following Python modules need to be installed:

    numpy / pandas / scikit-learn / xgboost

You need to place train.csv, test.csv, and submission.csv in the data/ directory,
or replace the following directory paths in conf/simple.conf with the actual paths to train.csv, test.csv, and submission.csv：

    train_filename=./data/train.csv
    test_filename=./data/test.csv
    submission_filename=./data/submission.csv

Then, run shelter.py in the bin/ directory:

    python bin/shelter.py
  
The following configurations in conf/simple.conf control the operations:

    do_train=1          # whether to train
    do_validation=1     # whether to perform validation
    do_search_parameter # whether to perform parameter search (needs corresponding parameter range specified in the code)
    do_test=1           # whether to test

Project file structure:

    bin/
      simple.py / shelter.py: related to the final version of the project results
    lib/
      *: related to the final version of the project results

    bin.bk/
      *_simple.py, model_average.py: historical attempts, including individual usage of random forest, KNN, XGBoost, and model fusion attempts


For more information about the project, please refer to [my blog](https://linpingta.github.io/blog/2016/07/10/kaggle-shelter-animal-outcome/).

I have kept all historical versions in the bin.bk/ directory without making any formatting corrections. The main reason is that I didn't have a suitable model training template in the early stages to try different model methods. After unifying various model methods later on, I developed [my model trainer template method](https://github.com/linpingta/tools/tree/master/model_trainer), which was used in the final version submission.
关于项目的介绍请见我的[博客](https://linpingta.github.io/blog/2016/07/10/kaggle-shelter-animal-outcome/)。
  
