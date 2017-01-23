#-*- coding: utf-8 -*-
#!/usr/bin/env python
# vim: set bg=dark noet ts=4 sw=4 fdm=indent :

""" lr model"""
__author__ = 'chutong'


from sklearn.linear_model import LogisticRegression
from joblib_loader import TsJoblibModelLoader


class LRModel(TsJoblibModelLoader):
	""" LR Model
	"""
	def __init__(self, model_conf):
		super(LRModel, self).__init__(model_conf)

		self.C = model_conf.getfloat('lr_classifier', 'C')

	def get_model(self, splited_key, logger):
		return LogisticRegression(C=self.C)


