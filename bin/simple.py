#-*- coding: utf-8 -*-

''' simple test for shelter animal outcome'''
__author__ = 'chutong'

import sys, os
import logging
import ConfigParser
import time, datetime
import re
try:
    import cPickle as pickle
except:
    import pickle

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.externals import joblib
from sklearn import cross_validation


class SimpleModel(object):
    ''' Simple Classifier 
    '''
    def __init__(self, conf):
        self.train_filename = conf.get('simple_model', 'train_filename')
        self.test_filename = conf.get('simple_model', 'test_filename')
        self.submission_filename = conf.get('simple_model', 'submission_filename')
	self.model_filename = conf.get('simple_model', 'model_filename')
	self.vc_filename = conf.get('simple_model', 'vc_filename')
	self.le_filename = conf.get('simple_model', 'le_filename')
	self.do_train = conf.getboolean('simple_model', 'do_train')
	self.do_validate = conf.getboolean('simple_model', 'do_validate')
	self.do_test = conf.getboolean('simple_model', 'do_test')

    def _load_csv_data(self, filename, logger):
        csv_data = None
        try:
            csv_data = pd.read_csv(filename)
	except Exception as e:
	    logger.exception(e)
	finally:
	    return csv_data

    def _dump_csv_data(self, df, filename, logger):
	try:
	    now = time.localtime()
	    postfix_time = time.strftime('%Y%m%d%H%M', now)
	    postfix_filename = '.'.join([filename, str(postfix_time)])
            df.to_csv(filename)
            df.to_csv(postfix_filename)
	except Exception as e:
	    logger.exception(e)

    def _clean_data(self, data, logger):
	data.fillna(0)
	data = data.apply(lambda x: x.fillna(x.value_counts().index[0]))
	return data

    def _transfer_age_info(self, age):
	#print 'age ', age
	age_in_day = 1
	year = re.search('([0-9]*) year', age)
	if year:
	    year_num = year.group(1)
	    age_in_day = 365 * int(year_num)
	month = re.search('([0-9]*) month', age)
	if month:
	    month_num = month.group(1)
	    age_in_day = 30 * int(month_num)
	week = re.search('([0-9]*) week', age)
	if week:
	    week_num = week.group(1)
	    age_in_day = 7 * int(week_num)
	day = re.search('([0-9]*) day', age)
	if day:
	    day_num = day.group(1)
	    age_in_day = int(day_num) 

	# manual split age
	age_list = [0, 30, 365 * 0.5, 365, 365 * 2, 365 * 5, 365 * 10]
	for idx, start_age in enumerate(age_list):
	    if age_in_day < start_age:
		return 'age' + str(idx)
	    
	return 'age' + str(len(age_list))

    def _transfer_age_infos(self, origin_age_info):
	#print 'origin_age_info ', origin_age_info
	new_age_info = origin_age_info.apply(self._transfer_age_info)
	#print 'new_age_info ', new_age_info
	return new_age_info

    def _encode_y(self, y, logger):
	le_y = LabelEncoder()
	le_y.fit(y)
	encode_y = le_y.transform(y)
	return (encode_y, le_y)

    def _transfer_data_to_model(self, data, logger):
	''' extract data from DataFrame'''

	# encode y
	(encode_y, le_y) = self._encode_y(data['OutcomeType'].values,logger)
	#print encode_y

	# encode x
	new_age_info = self._transfer_age_infos(data['AgeuponOutcome'])
	data['EncodeAgeuponOutcome'] = new_age_info
	#print np.isnan(data.any())
	#print np.isfinite(data.all())

	df = data.drop(['AnimalID', 'Name', 'DateTime', 'OutcomeType', 'OutcomeSubtype', 'AgeuponOutcome'], 1)
	#print df.isnull().sum()
	#print pd.isnull(df).any(1).nonzero()[0]

	x = df.T.to_dict().values()
	#print x
	vectorizer_x = DV(sparse=False)
	encode_x = vectorizer_x.fit_transform(x)
	#print encode_x
	return (encode_x, encode_y, vectorizer_x, le_y)

    def _get_model(self, logger):
	return None

    def run(self, now, logger):

	if self.do_train:
		# load train data
		# data cleaning
		train_data = self._load_csv_data(self.train_filename, logger)
		cleaned_train_data = self._clean_data(train_data, logger)

		# transfer to model format
		(train_x, train_y, vectorizer_x, le_y) = self._transfer_data_to_model(cleaned_train_data, logger)

		# select model
		clf = self._get_model(logger)
		clf.fit(train_x, train_y)
		if not clf:
		    logger.error('model not defined, no more train, quit')
		    return
		print clf
		joblib.dump(clf, self.model_filename)
		joblib.dump(vectorizer_x, self.vc_filename)
		joblib.dump(le_y, self.le_filename)
	else:
		clf = joblib.load(self.model_filename)
		vectorizer_x = joblib.load(self.vc_filename)
		le_y = joblib.load(self.le_filename)

	if self.do_validate:
		train_data = self._load_csv_data(self.train_filename, logger)
		cleaned_train_data = self._clean_data(train_data, logger)

		# transfer to model format
		(train_x, train_y, vectorizer_x, le_y) = self._transfer_data_to_model(cleaned_train_data, logger)
		#scores = cross_validation.cross_val_score(clf, train_x, train_y, pre_dispatch=1)
		#print 'accrucy mean %0.2f +/- %0.2f' % (scores.mean(), scores.std()*2)
		
	if self.do_test:
		# load test data
		# data cleaning
		test_data = self._load_csv_data(self.test_filename, logger)
		cleaned_test_data = self._clean_data(test_data, logger)
		test_x = cleaned_test_data.T.to_dict().values()
		encode_test_x = vectorizer_x.transform(test_x)
		#print 'encode_test_x ', encode_test_x
		#for idx, row in cleaned_test_data.iterrows():
		#    test_x = row.to_dict().values()
		#    encode_test_x = vectorizer.transform(test_x)
		#    print encode_test_x

		## perdict
		encode_test_y = clf.predict_proba(encode_test_x)
		#print 'encode_test_y ', encode_test_y
		#label_y = le_y.inverse_transform(encode_test_y)
		#print 'label_y ', label_y 
		output_y_list = []
		position_match_dict = {
		    0:'Adoption',
		    1:'Died',
		    2:'Euthanasia',
		    3:'Return_to_owner',
		    4:'Transfer'
		}
		for encode_test_yy in encode_test_y:
		    single_y_dict = {'Adoption':0,
			'Died':0,
			'Euthanasia':0,
			'Return_to_owner':0,
			'Transfer':0
		    } 
		    #if single_y not in single_y_dict:
		    #    logger.error('idx[%d] single_y %s not defined in existing labels' % (idx, single_y))
		    #    continue
		    #single_y_dict[single_y] = 1
		    for idx, single_y in enumerate(encode_test_yy):
			single_y_dict[position_match_dict[idx]] = single_y
		    output_y_list.append(single_y_dict)
		output_y = pd.DataFrame(output_y_list)
		output_y.index += 1
		self._dump_csv_data(output_y, self.submission_filename, logger)
	

class TsRandomForestClassfier(SimpleModel):
    def __init__(self, conf):
	super(TsRandomForestClassfier, self).__init__(conf)

	self.sub_tree_num = conf.getint('random_forest_classifier', 'sub_tree_num')

    def _get_model(self, logger):
	return RandomForestClassifier(n_estimators=self.sub_tree_num)


class TsRandomForestRegressor(SimpleModel):
    def __init__(self, conf):
	super(TsRandomForestRegressor, self).__init__(conf)

	self.sub_tree_num = conf.getint('random_forest_regressor', 'sub_tree_num')

    def _get_model(self, logger):
	return RandomForestRegressor(n_estimators=self.sub_tree_num)


if __name__ == '__main__':

    basepath = os.path.abspath(os.path.dirname(sys.path[0]))
    confpath = os.path.join(basepath, 'conf/simple_test.conf')
    conf = ConfigParser.RawConfigParser()
    conf.read(confpath)

    logging.basicConfig(filename=os.path.join(basepath, 'logs/simple_test.log'), level=logging.DEBUG,
        format = '[%(filename)s:%(lineno)s - %(funcName)s %(asctime)s;%(levelname)s] %(message)s',
        datefmt = '%a, %d %b %Y %H:%M:%S'
        )
    logger = logging.getLogger('simple_test')

    now = time.localtime()
    TsRandomForestClassfier(conf).run(now, logger)


