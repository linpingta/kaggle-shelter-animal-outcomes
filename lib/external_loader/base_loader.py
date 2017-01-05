#-*- coding: utf-8 -*-
#!/usr/bin/env python
# vim: set bg=dark noet ts=4 sw=4 fdm=indent :

""" Base data loader"""
__author__ = 'chutong'


from abc import ABCMeta, abstractmethod


class TsLoader(object):
	""" Abstract data loader 
	"""
	__meta__ = ABCMeta

	def __init__(self):
		pass

	@abstractmethod
	def load(self, filename, logger):
		pass

	@abstractmethod
	def dump(self, data, filename, logger):
		pass

