# -*- coding:utf-8 -*-
import sys
from struct import pack

def error(msg, code=1):
	print(msg)
	sys.exit(code)

def lsplit(s, sep = ' '):
	l = s.split(sep)
	return [l[0], sep.join(l[1:])]

def read(filename):
	file = open(filename, 'r', encoding='utf-8')
	cnt = file.read()
	file.close()
	return cnt

def bread(filename):
	file = open(filename, 'rb')
	cnt = file.read()
	file.close()
	return cnt

def write(content, filename):
	file = open(filename, 'w', encoding='utf-8')
	file.write(content)
	file.close()
	
def bwrite(content, filename):
	file = open(filename, 'wb')
	file.write(content)
	file.close()

def toint(s):
	if type(s) == int:
		return s
	try:
		return eval(s)
	except:
		pass
	s = s.strip('()')
	indic = s[0]
	n = s[1:]
	if indic == '$':
		return int(n, 16)
	elif indic == '#':
		return int(n)
	elif indic == '&':
		return int(n, 8)
	elif indic == '%':
		return int(n, 2)
	elif indic == '0':
		indic = n[0]
		n = n[1:]
		if indic == 'x':
			return int(n, 16)
		elif indic == 'b':
			return int(n, 2)
		elif indic == 'o':
			return int(n, 8)
		else:
			return int(indic + n)
	else:
		return int(indic + n)

def u8(s):
	return pack('<B', toint(s))

def u16(s):
	return pack('<H', toint(s))

def u16b(s):
	return pack('>H', toint(s))

def i8(s):
	return pack('<b', toint(s))

def isnum(s):
	return s.startswith(('$', '#', '%', '&', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))

def isptr(s):
	return s.startswith('(') and s.endswith(')')

def isreg(s):
	return s in ('a', 'b', 'c', 'd', 'e', 'f', 'h', 'l', 'hl', 'af', 'bc', 'de', 'sp', '(hl)', '(hl+)', '(hl-)', '(bc)', '(de)')

def isreg8(s):
	return s in ('a', 'b', 'c', 'd', 'e', 'f', '(hl)')

def isreg16(s):
	return s in ('af', 'bc', 'de', 'hl', 'sp')

def isregpoint(s):
	return s in ('(af)', '(bc)', '(de)', '(hl)', '(hl+)', '(hl-)')

class FakeFile (object):
	def __init__(self, data, byteorder='@'):
		self.ptr = 0
		self.data = bytearray(data)
		self.tp = TypeReader()
		self.tp.byteorder = byteorder
	def read(self, num = None):
		if num is None:
			return self.data[self.ptr:]
			self.ptr = len(data)
		else:
			if self.ptr + num >= len(self.data):
				raise IOError('Not as many data to read')
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
			if self.ptr + num >= len(data):
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
