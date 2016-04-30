#-*- coding: utf-8 -*-

''' simple test for shelter animal outcome'''
__author__ = 'chutong'

import sys, os
import logging
import ConfigParser
import time, datetime

import pandas as pd
from sklearn.ensemble import RandomForestClassfier


class SimpleClassifier(object):
    ''' Simple Classifier 
    '''
    def __init__(self, conf):
        self.train_filename = conf.get('simple_classifier', 'train_filename')
        self.test_filename = conf.get('simple_classifier', 'test_filename')
        self.submission_filename = conf.get('simple_classifier', 'submission_filename')

    def _load_csv_data(self, filename, logger):
        csv_data = None
        try:
            pd.read_csv(filename)
	except Exception as e:
	    logger.exception(e)
	finally:
	    return csv_data

    def _dump_csv_data(self, df, filename):
	try:
            df.to_csv(filename)
	except Exception as e:
	    logger.exception(e)

    def _clean_data(self, data, logger):
	return data

    def _transfer_data_to_model(self, data, logger):
	x = []
	y = []
	return (x, y)

    def _get_model(self, logger):
	return None

    def run(self, now, logger):

	# load train data
	# data cleaning
        train_data = self._load_csv_data(self.train_filename, logger)
	cleaned_train_data = self._clean_data(train_data, logger)

	# transfer to model format
	(train_x, train_y) = self._transfer_data_to_model(cleaned_train_data, logger)

 	# select model
	clf = self._get_model(logger)
	if not clf:
	    logger.error('model not defined, no more train, quit')

	# train model
	clf.fit(train_x, train_y)

	# load test data
	# data cleaning
	test_data = self._load_csv_data(self.test_filename, logger)
	cleaned_test_data = self._clean_data(test_data, logger)

	# perdict
	output = clf.predict(cleaned_test_data)
	self._dump_csv_data(output, self.submission_filename, logger)
	

class RandomForestClassfier(SimpleClassifier):
    def __init__(self, conf):
	super(RandomForestClassfier, self).__init__(conf)

	self.sub_tree_num = conf.getint('random_forest_classifier', 'sub_tree_num')

    def _get_model(self, logger):
	return RandomForestClassfier(n_estimator=self.sub_tree_num)

if __name__ == '__main__':

    basepath = os.path.abspath(os.path.dirname(sys.path[0]))
    confpath = os.path.join(basepath, 'conf/simple_test.conf')
    conf = ConfigParser.RawConfigParser()
    conf.read(confpath)

    logging.basicConfig(filename=os.path.join(basepath, 'logs.simple_test.log', level=logging.DEBUG,
        format='[%(filename)s:%(lineno)s - %(funcName)s %(asctime)s;%(levelname)s] %(message)s]',
        datefmt='%a, %d %b %Y %H:%M:%S'
    )
    logger = logging.getLogger('simple_test')

    now = time.localtime()
    SimpleClassifier(conf).run(now, logger)


