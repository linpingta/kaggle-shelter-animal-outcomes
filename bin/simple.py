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
	self.store_model = conf.getboolean('simple_model', 'store_model')

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
	age_list = [0, 30, 90, 365 * 0.5, 365, 365 * 2, 365 * 5, 365 * 10]
	for idx, start_age in enumerate(age_list):
	    if age_in_day < start_age:
		return 'age' + str(idx)
	    
	return 'age' + str(len(age_list))
	#return age_in_day

    def _transfer_age_infos(self, origin_age_info):
	#print 'origin_age_info ', origin_age_info
	new_age_info = origin_age_info.apply(self._transfer_age_info)
	#print 'new_age_info ', new_age_info
	return new_age_info

    def _transfer_year_info(self, animal_time):
	s = time.strptime(animal_time, '%Y-%m-%d %H:%M:%S')
	return s.tm_year

    def _transfer_month_info(self, animal_time):
	s = time.strptime(animal_time, '%Y-%m-%d %H:%M:%S')
	return s.tm_mon

    def _transfer_weekday_info(self, animal_time):
	s = time.strptime(animal_time, '%Y-%m-%d %H:%M:%S')
	return s.tm_wday
	#if s.tm_wday >= 5:
	#	return 'weekend'
	#else:
	#	return 'weekday'

    def _transfer_hour_info(self, animal_time):
	s = time.strptime(animal_time, '%Y-%m-%d %H:%M:%S')
	return s.tm_hour
	#if 0 <= s.tm_hour < 8:
	#	return 'hour1'
	#elif s.tm_hour < 18:
	#	return 'hour2'
	#else:
	#	return 'hour3'

    def _transfer_time_infos(self, origin_time_info):
	#print 'origin_time_info ', origin_time_info
	new_year_info = origin_time_info.apply(self._transfer_year_info)
	new_month_info = origin_time_info.apply(self._transfer_month_info)
	new_weekday_info = origin_time_info.apply(self._transfer_weekday_info)
	new_hour_info = origin_time_info.apply(self._transfer_hour_info)
	return (new_year_info, new_month_info, new_weekday_info, new_hour_info)

    def _encode_y(self, y, logger):
	le_y = LabelEncoder()
	le_y.fit(y)
	encode_y = le_y.transform(y)
	return (encode_y, le_y)

    def _transfer_data_to_model(self, data, animal, logger):
	''' extract data from DataFrame'''

	# encode y
	(encode_y, le_y) = self._encode_y(data['OutcomeType'].values,logger)
	#print encode_y

	# encode x
	if animal == 'Dog':
		new_age_info = self._transfer_age_infos(data['AgeuponOutcome'])
		data['EncodeAgeuponOutcome'] = new_age_info

		(year, month, weekday, hour) = self._transfer_time_infos(data['DateTime'])
		data['EncodeYear'] = year
		data['EncodeMonth'] = month 
		data['EncodeWeekday'] = weekday 
		data['EncodeHour'] = hour 

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

    def _transfer_test_data(self, cleaned_test_data, animal_dict, logger):
	test_x = cleaned_test_data.T.to_dict().values()
	encode_test_y = []
	for test_xx in test_x:
		if test_xx['AnimalType'] not in animal_dict:
			logger.error('%s is new type of animal, no model, skip' % test_xx['AnimalType'])
			continue

		clf = animal_dict[test_xx['AnimalType']]['clf']
		vectorizer_x = animal_dict[test_xx['AnimalType']]['vectorizer_x']
		le_y = animal_dict[test_xx['AnimalType']]['le_y']

		new_test_xx = test_xx.copy()
		if test_xx['AnimalType'] == 'Dog':
			new_test_xx['EncodeAgeuponOutcome'] = self._transfer_age_info(test_xx['AgeuponOutcome'])
			new_test_xx['EncodeYear'] = self._transfer_year_info(test_xx['DateTime'])
			new_test_xx['EncodeMonth'] = self._transfer_month_info(test_xx['DateTime'])
			new_test_xx['EncodeWeekday'] = self._transfer_weekday_info(test_xx['DateTime'])
			new_test_xx['EncodeHour'] = self._transfer_hour_info(test_xx['DateTime'])
		remove_attributes = ['AnimalID', 'Name', 'DateTime', 'OutcomeType', 'OutcomeSubtype', 'AgeuponOutcome']
		for remove_attribute in remove_attributes:
			new_test_xx.pop(remove_attribute, None)
		#print new_test_xx

		encode_test_xx = vectorizer_x.transform(new_test_xx)
		#print 'encode_test_x ', encode_test_x
		#for idx, row in cleaned_test_data.iterrows():
		#    test_x = row.to_dict().values()
		#    encode_test_x = vectorizer.transform(test_x)
		#    print encode_test_x

		## perdict
		encode_test_yy = clf.predict_proba(encode_test_xx)
		#print 'encode_test_y ', encode_test_y
		#label_y = le_y.inverse_transform(encode_test_y)
		#print 'label_y ', label_y 
		encode_test_y.append(encode_test_yy[0])

	return encode_test_y

    def _output_y(self, encode_test_y, position_match_dict, logger):
	output_y_list = []
	for encode_test_yy in encode_test_y:
	    #print encode_test_yy
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
	return output_y_list

    def run(self, now, logger):

	animals = ['Cat', 'Dog']
	animal_dict = {}

	if self.do_train:
		# load train data
		# data cleaning
		train_data = self._load_csv_data(self.train_filename, logger)
		cleaned_train_data = self._clean_data(train_data, logger)

		# split data based on animal type
		for animal in animals:
			print animal
			#print cleaned_train_data[cleaned_train_data['AnimalType']==animal]
			# transfer to model format
			animal_train_data = cleaned_train_data[cleaned_train_data['AnimalType']==animal].copy()
			(train_x, train_y, vectorizer_x, le_y) = self._transfer_data_to_model(animal_train_data, animal, logger)

			# select model
			clf = self._get_model(logger)
			clf.fit(train_x, train_y)
			if not clf:
			    logger.error('model not defined, no more train, quit')
			    return
			#print clf

			animal_dict[animal] = {'clf': clf,
			    'vectorizer_x': vectorizer_x,
			    'le_y': le_y
			}

			joblib.dump(clf, '.'.join([self.model_filename,animal]))
			joblib.dump(vectorizer_x, '.'.join([self.vc_filename,animal]))
			joblib.dump(le_y, '.'.join([self.le_filename,animal]))
			if self.store_model:
				postfix_time = time.strftime('%Y%m%d%H%M', now)
				model_postfix_filename = '.'.join([self.model_filename, animal, str(postfix_time)])
				joblib.dump(clf, model_postfix_filename)
				vc_postfix_filename = '.'.join([self.vc_filename, animal, str(postfix_time)])
				joblib.dump(vectorizer_x, vc_postfix_filename)
				le_postfix_filename = '.'.join([self.le_filename, animal, str(postfix_time)])
				joblib.dump(le_y, le_postfix_filename)

			if self.do_validate:
				scores = cross_validation.cross_val_score(clf, train_x, train_y, pre_dispatch=1, scoring='log_loss')
				print 'accrucy mean %0.2f +/- %0.2f' % (scores.mean(), scores.std()*2)
	else:
		for animal in animals:
			clf = joblib.load(self.model_filename)
			vectorizer_x = joblib.load(self.vc_filename)
			le_y = joblib.load(self.le_filename)

			animal_dict[animal] = {'clf': clf,
			    'vectorizer_x': vectorizer_x,
			    'le_y': le_y
			}

	#if self.do_validate:
	#	train_data = self._load_csv_data(self.train_filename, logger)
	#	cleaned_train_data = self._clean_data(train_data, logger)

	#	# transfer to model format
	#	(train_x, train_y, vectorizer_x, le_y) = self._transfer_data_to_model(cleaned_train_data, logger)
	#	#scores = cross_validation.cross_val_score(clf, train_x, train_y, pre_dispatch=1)
	#	#print 'accrucy mean %0.2f +/- %0.2f' % (scores.mean(), scores.std()*2)
		
	if self.do_test:
		# load test data
		# data cleaning
		test_data = self._load_csv_data(self.test_filename, logger)
		cleaned_test_data = self._clean_data(test_data, logger)
		
		position_match_dict = {
		    0:'Adoption',
		    1:'Died',
		    2:'Euthanasia',
		    3:'Return_to_owner',
		    4:'Transfer'
		}
		encode_test_y = self._transfer_test_data(cleaned_test_data, animal_dict, logger)
    		output_y_list = self._output_y(encode_test_y, position_match_dict, logger)
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


