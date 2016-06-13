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

from operator import itemgetter
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

backup_color_list = ['White', 'Green', 'Blue', 'Black', 'Pink', 'Red', 'Brown', 'Orange', 'Yellow', 'Silver', 'Gold', 'Gray', 'Tan']


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
	self.intake_filename = conf.get('simple_model', 'intake_filename')
	self.do_train = conf.getboolean('simple_model', 'do_train')
	self.do_search_parameter = conf.getboolean('simple_model', 'do_search_parameter')
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
	#data.fillna(0)
	#data = data.apply(lambda x: x.fillna(x.value_counts().index[0]))
	#data = data[np.isfinite(data['SexuponOutcome'])]
	return data

    def _transfer_age_info(self, age):
	if (not age) or (age is np.nan):
		return 'age0'

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
		return 'age' + str(idx + 1)
	    
	return 'age' + str(len(age_list) + 1)
	#return age_in_day

    def _transfer_age_infos(self, origin_age_info):
	#print 'origin_age_info ', origin_age_info
	new_age_info = origin_age_info.apply(self._transfer_age_info)
	#print 'new_age_info ', new_age_info
	return new_age_info

    def _transfer_mix_info(self, breed):
	#return 'Mix' if ('Mix' in breed or '/' in breed) else 'UnMix'
	return 'Mix' if 'Mix' in breed else 'UnMix'
	#return len(breed.split('/')) + 1

    def _transfer_mix_infos(self, origin_breed_info):
	breed = origin_breed_info.apply(self._transfer_mix_info)
	return breed

    def _transfer_color_count_info(self, color):
	return len(color.split('/')) + 1

    def _transfer_color_count_infos(self, origin_color_info):
	color_count = origin_color_info.apply(self._transfer_color_count_info)
	return color_count

    def _transfer_species_info(self, color):
	pattern_list = []
	s = color.split(' ')
	total_elem = []
	for elem in s:
		total_elem.extend(elem.split('/'))
	for elem in total_elem:
		if elem not in backup_color_list:
			pattern_list.append(elem)
	if not pattern_list:
		return 'no_pattern'
	else:
		return '_'.join(pattern_list)

    def _transfer_species_infos(self, origin_color_info):
	species = origin_color_info.apply(self._transfer_species_info)
	return species

    def _transfer_color_info(self, color):
	s = color.split(' ')
	if len(s) >= 2:
		return s[0]
	else:
		for backup_color in backup_color_list:
			if backup_color in s[0]:
				return s[0]
		return color
		
	#color_list = []
	#total_elem = []
	#for elem in s:
	#	total_elem.extend(elem.split('/'))
	#for elem in total_elem:
	#	if elem in backup_color_list:
	#		color_list.append(elem)
	#	
	#if not color_list:
	#	return 'no_color'
	#else:
	#	return '_'.join(color_list)

    def _transfer_color_infos(self, origin_color_info):
	color = origin_color_info.apply(self._transfer_color_info)
	return color

    def _transfer_color_type_info(self, color, color_type):
	if color_type in color:
		if ' ' in color:
			return 0.5
		else:
			return 1
	else:
		return 0
	#return color_type in color

    def _transfer_color_type_infos(self, origin_color_info, color_type):
	has_color = origin_color_info.apply(self._transfer_color_type_info, args=(color_type,))
	return has_color

    def _transfer_breed_type_info(self, breed, breed_type):
	#if breed_type in breed:
	#	if '/' in breed:
	#		return 0.5
	#	else:
	#		return 1
	#else:
	#	return 0
	return breed_type in breed

    def _transfer_breed_type_infos(self, origin_breed_info, breed_type):
	has_breed = origin_breed_info.apply(self._transfer_breed_type_info, args=(breed_type,))
	return has_breed

    def _transfer_breed_info(self, breed):
	if 'Mix' in breed:
		return breed.replace(' Mix', '')
	elif '/' in breed:
		return breed.split('/')[0]
	else:
		return breed

    def _transfer_breed_infos(self, origin_breed_info):
	breed = origin_breed_info.apply(self._transfer_breed_info)
	return breed

    def _transfer_sex_info(self, sex):
	if sex is np.nan:
		return 'Unknown'

	choices = ['Female', 'Male']
	for choice in choices:
		if choice in sex:
			return choice
	return 'Unknown'

    def _transfer_sex_infos(self, origin_sex_info):
	sex = origin_sex_info.apply(self._transfer_sex_info)
	return sex

    def _transfer_intact_info(self, sex):
	if sex is np.nan:
		return 'Unknown'

	#choices = ['Intact', 'Neutered', 'Spayed']
	choices = ['Intact']
	for choice in choices:
		if choice in sex:
			return choice
	return 'Unknown'

    def _transfer_intact_infos(self, origin_sex_info):
	intact = origin_sex_info.apply(self._transfer_intact_info)
	return intact

    def _transfer_name_info(self, name):
	#return True if name else False
	#return 'HasName' if name else 'HasNotName'
	if name is np.nan:
		return False
	else:
		return True
	#return len(name) > 0

    def _transfer_name_infos(self, origin_name_info):
	#print 'origin_name_info', origin_name_info
	has_name = origin_name_info.apply(self._transfer_name_info)
	return has_name

    def _transfer_year_info(self, animal_time):
	s = time.strptime(animal_time, '%Y-%m-%d %H:%M:%S')
	return s.tm_year

    def _transfer_month_info(self, animal_time):
	s = time.strptime(animal_time, '%Y-%m-%d %H:%M:%S')
	return s.tm_mon

    def _transfer_weekday_info(self, animal_time):
	s = time.strptime(animal_time, '%Y-%m-%d %H:%M:%S')
	return s.tm_wday
	#if 5 <= s.tm_wday <= 6:
	#	return 'weekend'
	#else:
	#	return 'weekday'

    def _transfer_hour_info(self, animal_time):
	s = time.strptime(animal_time, '%Y-%m-%d %H:%M:%S')
	#return s.tm_hour
	if 5 < s.tm_hour < 11:
		return 'hour1'
	elif 10 < s.tm_hour < 16:
		return 'hour2'
	elif 15 < s.tm_hour < 20:
		return 'hour3'
	else:
		return 'hour4'

    def _transfer_time_infos(self, origin_time_info):
	#print 'origin_time_info ', origin_time_info
	new_year_info = origin_time_info.apply(self._transfer_year_info)
	new_month_info = origin_time_info.apply(self._transfer_month_info)
	new_weekday_info = origin_time_info.apply(self._transfer_weekday_info)
	new_hour_info = origin_time_info.apply(self._transfer_hour_info)
	return (new_year_info, new_month_info, new_weekday_info, new_hour_info)

    def _transfer_intake_location(self, animal_id, intake_df):
	animal_info = intake_df[intake_df['Animal ID'] == animal_id]
	if not animal_info.empty:
		return animal_info['Found Location']
	else:
		return 'Unknown'

    def _transfer_intake_type(self, animal_id, intake_df):
	animal_info = intake_df[intake_df['Animal ID'] == animal_id]
	animal_intake_type_setting = ['Euthanasia Request', 'Owner Surrender', 'Public Assist', 'Stray', 'Wildlife']
	if animal_info.empty:
		return 'Unknown'
	animal_intake_type = animal_info['Intake Type'].values[0].strip()
	#print animal_intake_type
	if animal_intake_type in animal_intake_type_setting:
		return animal_intake_type
	else:
		return 'Unknown'

    def _transfer_intake_condition(self, animal_id, intake_df):
	animal_info = intake_df[intake_df['Animal ID'] == animal_id]
	animal_intake_condition_setting = ['Aged', 'Feral', 'Injured', 'Normal', 'Nursing', 'Other', 'Pregnant', 'Sick']
	if animal_info.empty:
		return 'Unknown'
	animal_intake_condition = animal_info['Intake Condition'].values[0].strip()
	#print animal_intake_condition
	if animal_intake_condition in animal_intake_condition_setting:
		return animal_intake_condition
	else:
		return 'Unknown'

    def _transfer_intake_infos(self, origin_animal_info, intake_df):
	intake_location = origin_animal_info.apply(self._transfer_intake_location, args=(intake_df,))
	intake_type = origin_animal_info.apply(self._transfer_intake_type, args=(intake_df,))
	intake_condition = origin_animal_info.apply(self._transfer_intake_condition, args=(intake_df,))
	return (intake_location, intake_type, intake_condition)

    def _encode_y(self, y, logger):
	le_y = LabelEncoder()
	le_y.fit(y)
	encode_y = le_y.transform(y)
	return (encode_y, le_y)

    def _transfer_data_to_model(self, data, animal, total_info, logger):
	''' extract data from DataFrame'''
	total_breed = total_info[0]
	total_color = total_info[1]
	intake_df = total_info[2]

	# encode y
	(encode_y, le_y) = self._encode_y(data['OutcomeType'].values,logger)
	#print encode_y

	# encode x
	#if animal in ('Dog', 'All'):
	if True:
	#if False:
		new_age_info = self._transfer_age_infos(data['AgeuponOutcome'])
		data['EncodeAgeuponOutcome'] = new_age_info

		(year, month, weekday, hour) = self._transfer_time_infos(data['DateTime'])
		data['EncodeYear'] = year
		data['EncodeMonth'] = month 
		data['EncodeWeekday'] = weekday 
		data['EncodeHour'] = hour 
		drop_list = ['AnimalID', 'Name', 'DateTime', 'OutcomeType', 'OutcomeSubtype', 'AgeuponOutcome', 'SexuponOutcome', 'Breed', 'Color']
		#drop_list = ['AnimalID', 'Name', 'DateTime', 'OutcomeType', 'OutcomeSubtype', 'AgeuponOutcome', 'SexuponOutcome', 'Breed']
		#drop_list = ['AnimalID', 'Name', 'DateTime', 'OutcomeType', 'OutcomeSubtype', 'AgeuponOutcome', 'SexuponOutcome']

	data['HasName'] = self._transfer_name_infos(data['Name'])
	data['Sex'] = self._transfer_sex_infos(data['SexuponOutcome'])
	data['Intact'] = self._transfer_intact_infos(data['SexuponOutcome'])
	data['IsMix'] = self._transfer_mix_infos(data['Breed'])
	#data['NewBreed'] = self._transfer_breed_infos(data['Breed'])
	#data['Species'] = self._transfer_species_infos(data['Color'])
	#data['NewColor'] = self._transfer_color_infos(data['Color'])
	data['ColorMix'] = self._transfer_color_count_infos(data['Color'])
	for breed_type in total_breed:
		data['Breed%s' % breed_type] = self._transfer_breed_type_infos(data['Breed'], breed_type)
	for color_type in total_color:
		data['Color%s' % color_type] = self._transfer_color_type_infos(data['Color'], color_type)
	(found_location, intake_type, intake_condition) = self._transfer_intake_infos(data['AnimalID'], intake_df)
	#data['FoundLocation'] = found_location
	#data['IntakeType'] = intake_type
	#data['IntakeCondition'] = intake_condition

	#print np.isnan(data.any())
	#print np.isfinite(data.all())
	df = data.drop(drop_list, 1)
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

    def _get_grid_search_model(self, logger):
	return None

    def _transfer_test_data(self, cleaned_test_data, animal_dict, total_info, logger):
	total_breed = total_info[0]
	total_color = total_info[1]
	intake_df = total_info[2]

	test_x = cleaned_test_data.T.to_dict().values()
	encode_test_y = []
	for test_xx in test_x:
		if test_xx['AnimalType'] not in animal_dict:
			logger.error('%s is new type of animal, no model, skip' % test_xx['AnimalType'])
			continue

		if ('All' in animal_dict) and (test_xx['AnimalType'] == 'Dog'):
			clf = animal_dict['All']['clf']
			vectorizer_x = animal_dict['All']['vectorizer_x']
			le_y = animal_dict['All']['le_y']
		else:
			clf = animal_dict[test_xx['AnimalType']]['clf']
			vectorizer_x = animal_dict[test_xx['AnimalType']]['vectorizer_x']
			le_y = animal_dict[test_xx['AnimalType']]['le_y']

		new_test_xx = test_xx.copy()
		#if test_xx['AnimalType'] == 'Dog':
		if True:
		#if False:
			new_test_xx['EncodeAgeuponOutcome'] = self._transfer_age_info(test_xx['AgeuponOutcome'])
			new_test_xx['EncodeYear'] = self._transfer_year_info(test_xx['DateTime'])
			new_test_xx['EncodeMonth'] = self._transfer_month_info(test_xx['DateTime'])
			new_test_xx['EncodeWeekday'] = self._transfer_weekday_info(test_xx['DateTime'])
			new_test_xx['EncodeHour'] = self._transfer_hour_info(test_xx['DateTime'])
			#remove_attributes = ['AnimalID', 'Name', 'DateTime', 'OutcomeType', 'OutcomeSubtype', 'AgeuponOutcome', 'SexuponOutcome']
			remove_attributes = ['AnimalID', 'Name', 'DateTime', 'OutcomeType', 'OutcomeSubtype', 'AgeuponOutcome', 'SexuponOutcome', 'Breed', 'Color']
		else:
			remove_attributes = ['AnimalID', 'Name', 'DateTime', 'OutcomeType', 'OutcomeSubtype', 'SexuponOutcome']
			#remove_attributes = ['AnimalID', 'Name', 'DateTime', 'OutcomeType', 'OutcomeSubtype', 'SexuponOutcome', 'Breed', 'Color']

		new_test_xx['HasName'] = self._transfer_name_info(test_xx['Name'])
		new_test_xx['Sex'] = self._transfer_sex_info(test_xx['SexuponOutcome'])
		new_test_xx['Intact'] = self._transfer_intact_info(test_xx['SexuponOutcome'])
		new_test_xx['IsMix'] = self._transfer_mix_info(test_xx['Breed'])
		#new_test_xx['NewBreed'] = self._transfer_breed_info(test_xx['Breed'])
		#new_test_xx['Species'] = self._transfer_species_info(test_xx['Color'])
		#new_test_xx['NewColor'] = self._transfer_color_info(test_xx['Color'])
		new_test_xx['ColorMix'] = self._transfer_color_count_info(test_xx['Color'])
		for breed_type in total_breed:
			new_test_xx['Breed%s' % breed_type] = self._transfer_breed_type_info(test_xx['Breed'], breed_type)
		for color_type in total_color:
			new_test_xx['Color%s' % color_type] = self._transfer_color_type_info(test_xx['Color'], color_type)

		#new_test_xx['FoundLocation'] = self._transfer_intake_location(test_xx['AnimalID'], intake_df)
		#new_test_xx['IntakeType'] = self._transfer_intake_type(test_xx['AnimalID'], intake_df)
		#new_test_xx['IntakeCondition'] = self._transfer_intake_condition(test_xx['AnimalID'], intake_df)

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

    def report(self, grid_scores, n_top, logger):
        top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
        for i, score in enumerate(top_scores):
            print("Model with rank: {0}".format(i + 1))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  score.mean_validation_score,
                  np.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters))
            print("")
            logger.debug("Model with rank: {0}".format(i + 1))
            logger.debug("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  score.mean_validation_score,
                  np.std(score.cv_validation_scores)))
            logger.debug("Parameters: {0}".format(score.parameters))

    def run(self, now, logger):

	animals = ['All', 'Cat', 'Dog']
	#animals = ['Cat', 'Dog']
	#animals = ['All']
	animal_dict = {}

	# load train data
	# data cleaning
	train_data = self._load_csv_data(self.train_filename, logger)
	cleaned_train_data = self._clean_data(train_data, logger)
	
	# load test data
	# data cleaning
	test_data = self._load_csv_data(self.test_filename, logger)
	cleaned_test_data = self._clean_data(test_data, logger)
		
	total_info = []

	train_breed = cleaned_train_data['Breed'].unique()
	new_train_breed = []
	for breed in train_breed:
		tmp_breed = breed.replace(' Mix', '')
		new_train_breed.extend(tmp_breed.split('/'))
	test_breed = cleaned_test_data['Breed'].unique()
	new_test_breed = []
	for breed in test_breed:
		tmp_breed = breed.replace(' Mix', '')
		new_test_breed.extend(tmp_breed.split('/'))
	total_breed = list(set(new_train_breed) | set(new_test_breed))
	total_info.append(total_breed)

	train_color = cleaned_train_data['Color'].unique()
	new_train_color = []
	for color in train_color:
		new_train_color.extend(color.split(' '))
	test_color = cleaned_test_data['Color'].unique()
	new_test_color = []
	for color in test_color:
		new_test_color.extend(color.split(' '))
	total_color = list(set(new_train_color) | set(new_test_color))
	total_info.append(total_color)

	intake_df = self._load_csv_data(self.intake_filename, logger)
	total_info.append(intake_df)

	if self.do_train:

		# split data based on animal type
		for animal in animals:
			print animal, 'train'
			#if animal == 'Cat':
			#   continue
			#print cleaned_train_data[cleaned_train_data['AnimalType']==animal]
			# transfer to model format
			if animal == 'All':
				animal_train_data = cleaned_train_data.copy()
			else:
				animal_train_data = cleaned_train_data[cleaned_train_data['AnimalType']==animal].copy()
			(train_x, train_y, vectorizer_x, le_y) = self._transfer_data_to_model(animal_train_data, animal, total_info, logger)

			if self.do_search_parameter:
				# search momdel
				print 'search parameter'
				clf = self._get_grid_search_model(animal, logger)
				# xgboost parameter group1
				#param_grid = {"max_depth": range(9,15,2),
				#	'min_child_weight':range(5,10,2),
				#}
				# xgboost parameter group2
				#param_grid = {"subsample": [0.5, 0.7, 0.9],
				#	"colsample_bytree": [0.5, 0.7, 0.9]
				#}
				# xgboost parameter group3
				#param_grid = {"learning_rate": [0.05, 0.1, 0.15, 0.2, 0.5],
				#	"n_estimators": [100, 250, 500],
				#	#"max_depth": range(7,15,2),
				#}
				# xgboost parameter group1
				#param_grid = {
				#	'min_child_weight':range(1,5,2),
				#}
				# xgboost parameter group1
				#param_grid = {"learning_rate": [0.03, 0.04, 0.05],
				#}
				param_grid = {
					"colsample_bytree": [0.7, 0.8, 0.9]
				}
				# RF
				#param_grid = {"max_depth": range(3, 100, 5),
				#      "max_features": [1, 3, 10],
				#      "min_samples_split": [1, 3, 10],
				#      "min_samples_leaf": [1, 3, 10],
				#      "criterion": ["gini", "entropy"]}
				grid_search = GridSearchCV(clf, scoring="log_loss", param_grid=param_grid)
				grid_search.fit(train_x, train_y)
				self.report(grid_search.grid_scores_, 3, logger)
			else:
				print 'train model'
				# select model
				#clf = self._get_model(animal, logger)
				#clf.fit(train_x, train_y)
				clf = self._get_model(animal, logger)
				clf.fit(train_x, train_y, eval_metric='mlogloss')
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

				if self.do_validate:
					scores = cross_validation.cross_val_score(clf, train_x, train_y, pre_dispatch=1, scoring='log_loss')
					print 'accrucy mean %0.2f +/- %0.2f' % (scores.mean(), scores.std()*2)
					logger.info('animal %s accrucy mean %0.2f +/- %0.2f' % (animal, scores.mean(), scores.std()*2))

					if self.store_model:
						postfix_time = time.strftime('%Y%m%d%H%M', now)
						score_info = str(scores.mean())
						model_postfix_filename = '.'.join([self.model_filename, animal, str(postfix_time), score_info])
						joblib.dump(clf, model_postfix_filename)
						vc_postfix_filename = '.'.join([self.vc_filename, animal, str(postfix_time), score_info])
						joblib.dump(vectorizer_x, vc_postfix_filename)
						le_postfix_filename = '.'.join([self.le_filename, animal, str(postfix_time), score_info])
						joblib.dump(le_y, le_postfix_filename)
	else:
		for animal in animals:
			clf = joblib.load('.'.join([self.model_filename, animal]))
			vectorizer_x = joblib.load('.'.join([self.vc_filename, animal]))
			le_y = joblib.load('.'.join([self.le_filename, animal]))

			animal_dict[animal] = {'clf': clf,
			    'vectorizer_x': vectorizer_x,
			    'le_y': le_y
			}
		
	if self.do_test:
		position_match_dict = {
		    0:'Adoption',
		    1:'Died',
		    2:'Euthanasia',
		    3:'Return_to_owner',
		    4:'Transfer'
		}
		print 'test'
		encode_test_y = self._transfer_test_data(cleaned_test_data, animal_dict, total_info, logger)
    		output_y_list = self._output_y(encode_test_y, position_match_dict, logger)
		output_y = pd.DataFrame(output_y_list)
		output_y.index += 1
		self._dump_csv_data(output_y, self.submission_filename, logger)
	

class TsXgbClassifier(SimpleModel):
    def __init__(self, conf):
	super(TsXgbClassifier, self).__init__(conf)

    def _get_grid_search_model(self, animal, logger):
	return XGBClassifier(
		learning_rate=0.03,
		n_estimators=200,
		max_depth=11,
		subsample=0.75,
		#colsample_bytree=0.85,
		min_child_weight=3
		)

    def _get_model(self, animal, logger):
	return XGBClassifier(learning_rate=0.03,
		n_estimators=200,
		#silent=False,
		max_depth=11,
		min_child_weight=3,
		subsample=0.75,
		colsample_bytree=0.85,
		seed=121,
		objective='multi:softprob',
	)
	## best parameter
	#return XGBClassifier(learning_rate=0.1,
	#	n_estimators=100,
	#	#silent=False,
	#	max_depth=9,
	#	min_child_weight=5,
	#	subsample=0.7,
	#	colsample_bytree=0.9,
	#	#seed=121,
	#	objective='multi:softprob',
	#)


class TsRandomForestClassfier(SimpleModel):
    def __init__(self, conf):
	super(TsRandomForestClassfier, self).__init__(conf)

	self.sub_tree_num = conf.getint('random_forest_classifier', 'sub_tree_num')
	self.max_depth_num = conf.getint('random_forest_classifier', 'max_depth_num')
	self.cat_sub_tree_num = conf.getint('random_forest_classifier', 'cat_sub_tree_num')
	self.cat_max_depth_num = conf.getint('random_forest_classifier', 'cat_max_depth_num')
	self.dog_sub_tree_num = conf.getint('random_forest_classifier', 'dog_sub_tree_num')
	self.dog_max_depth_num = conf.getint('random_forest_classifier', 'dog_max_depth_num')

    def _get_model(self, animal, logger):
	
	if animal == 'Cat':
		return RandomForestClassifier(n_estimators=self.cat_sub_tree_num, max_depth=self.cat_max_depth_num)
	elif animal == 'Dog':
		return RandomForestClassifier(n_estimators=self.dog_sub_tree_num, max_depth=self.dog_max_depth_num)
	else:
		return RandomForestClassifier(n_estimators=self.sub_tree_num, max_depth=self.max_depth_num)
	#return RandomForestClassifier(n_estimators=self.sub_tree_num)

    def _get_grid_search_model(self, animal, logger):
	return RandomForestClassifier(n_estimators=self.sub_tree_num)


class TsRandomForestRegressor(SimpleModel):
    def __init__(self, conf):
	super(TsRandomForestRegressor, self).__init__(conf)

	self.sub_tree_num = conf.getint('random_forest_regressor', 'sub_tree_num')
	self.max_depth_num = conf.getint('random_forest_regressor', 'max_depth_num')

    def _get_model(self, logger):
	return RandomForestRegressor(n_estimators=self.sub_tree_num, max_depth=self.max_depth_num)

    def _get_grid_search_model(self, logger):
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
    #TsRandomForestClassfier(conf).run(now, logger)
    TsXgbClassifier(conf).run(now, logger)


