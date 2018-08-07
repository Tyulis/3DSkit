# -*- coding:utf-8 -*-
import os
import platform
try:
	import console
	has_console = True
except ImportError:
	has_console = False
from collections import OrderedDict
from util.rawutil import TypeReader

SWITCH_DEFAULT = object()


class ClsFunc (object):
	def __new__(cls, *args, **kwargs):
		self = object.__new__(cls)
		return self.main(*args, **kwargs)
	
	def main(self):
		return


class FreeObject (object):
	pass
	

class attrdict (OrderedDict):
	def __setattr__(self, attr, val):
		self[attr] = val
	
	def __getattr__(self, attr):
		return self[attr]


class FakeFile (object):
	def __init__(self, data, byteorder='@'):
		self.ptr = 0
		self.data = bytearray(data)
		self.tp = TypeReader()
		self.tp.byteorder = byteorder
	
	def read(self, num=None):
		if num is None:
			return self.data[self.ptr:]
			self.ptr = len(self.data)
		else:
			if self.ptr + num >= len(self.data):
				raise IOError('Not enough data to read')
			return self.data[self.ptr:self.ptr + num]
			self.ptr += num
	
	def write(self, data):
		self.data[self.ptr:self.ptr + len(data)] = data
		self.ptr += len(data)
	
	def seek(self, num, mode=0):
		if mode == 0:
			if num >= len(self.data):
				raise IOError('Tried to seek after the stream end')
			self.ptr = num
		elif mode == 1:
			if self.ptr + num >= len(self.data):
				raise IOError('Tried to seek after the stream end')
			self.ptr += num
		elif mode == 2:
			if len(self.data) - num < 0:
				raise IOError('Tried to seek before the stream start')
			self.ptr = len(self.data) - num
	
	def tell(self):
		return self.ptr
	
	def uint8(self):
		if self.ptr <= len(self.data) - 1:
			val, self.ptr = self.tp.uint8(self.data, self.ptr)
			return val
		else:
			raise IOError('Not enough data to unpack')
	
	def uint16(self):
		if self.ptr <= len(self.data) - 2:
			val, self.ptr = self.tp.uint16(self.data, self.ptr)
			return val
		else:
			raise IOError('Not enough data to unpack')
	
	def uint24(self):
		if self.ptr <= len(self.data) - 3:
			val, self.ptr = self.tp.uint24(self.data, self.ptr)
			return val
		else:
			raise IOError('Not enough data to unpack')
	
	def uint32(self):
		if self.ptr <= len(self.data) - 4:
			val, self.ptr = self.tp.uint32(self.data, self.ptr)
			return val
		else:
			raise IOError('Not enough data to unpack')
	
	def uint64(self):
		if self.ptr <= len(self.data) - 8:
			val, self.ptr = self.tp.uint64(self.data, self.ptr)
			return val
		else:
			raise IOError('Not enough data to unpack')
	
	def int8(self):
		if self.ptr <= len(self.data) - 1:
			val, self.ptr = self.tp.int8(self.data, self.ptr)
			return val
		else:
			raise IOError('Not enough data to unpack')
	
	def int16(self):
		if self.ptr <= len(self.data) - 2:
			val, self.ptr = self.tp.int16(self.data, self.ptr)
			return val
		else:
			raise IOError('Not enough data to unpack')
	
	def int24(self):
		if self.ptr <= len(self.data) - 3:
			val, self.ptr = self.tp.int24(self.data, self.ptr)
			return val
		else:
			raise IOError('Not enough data to unpack')
	
	def int32(self):
		if self.ptr <= len(self.data) - 4:
			val, self.ptr = self.tp.int32(self.data, self.ptr)
			return val
		else:
			raise IOError('Not enough data to unpack')
	
	def int64(self):
		if self.ptr <= len(self.data) - 8:
			val, self.ptr = self.tp.int64(self.data, self.ptr)
			return val
		else:
			raise IOError('Not enough data to unpack')


def clearconsole():
	if has_console:
		console.clear()
	elif platform.system() == 'Windows':
		os.system('cls')
	else:
		os.system('clear')


def getsup(lst, num):
	lst = sorted(lst)
	i = lst.index(num)
	return lst[i + 1]


def split(s, sep):
	if type(sep) in (str, bytes):
		return s.split(sep)
	elif type(sep) == int:
		return [s[i: i + sep] for i in range(0, len(s), sep)]


def toascii(string):
	'''Converts accentuated or special chatacters in a string in ascii characters'''
	s = string.replace('é', 'e').replace('è', 'e').replace('ê', 'e').replace('ë', 'e')
	s = s.replace('à', 'a').replace('â', 'a').replace('ę', 'e').replace('ė', 'e')
	s = s.replace('ē', 'e').replace('æ', 'ae').replace('á', 'a').replace('ä', 'a')
	s = s.replace('ã', 'a').replace('å', 'a').replace('ā', 'a').replace('ÿ', 'y')
	s = s.replace('û', 'u').replace('ù', 'u').replace('ü', 'u').replace('î', 'i')
	s = s.replace('ï', 'i').replace('ú', 'u').replace('ū', 'u').replace('ì', 'i')
	s = s.replace('í', 'i').replace('į', 'i').replace('ī', 'i').replace('ô', 'o')
	s = s.replace('œ', 'oe').replace('ö', 'o').replace('ø', 'o').replace('ç', 'c')
	s = s.replace('ñ', 'n').replace('ò', 'o').replace('ó', 'o').replace('õ', 'o')
	s = s.replace('ō', 'o').replace('º', 'o').replace('ć', 'c').replace('č', 'c')
	s = s.replace('ń', 'n').replace('€', 'E').replace('£', 'L').replace('¥', 'Y')
	s = s.replace('•', '.').replace('§', 'S').replace('¿', '?').replace('¡', '!')
	s = s.replace('«', '"').replace('»', '"').replace('„', '"').replace('“', '"')
	s = s.replace('”', '"')
	s = ''.join([c for c in s if ord(c) < 128])
	return s


def byterepr(s):
	final = repr(s)
	return final[2: -1]  #strips b''

def switch(value, args, callbacks):
	'''switch(value, (arg1, arg2), {
		value1: callback1,
		value2: callback2,
	})'''
	if value in callbacks.keys():
		return callbacks[value](*args)
	else:
		if SWITCH_DEFAULT in callbacks.keys():
			return callbacks[SWITCH_DEFAULT](*args)