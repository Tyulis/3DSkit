# -*- coding:utf-8 -*-
# rawutil.py
# A single-file pure-python module to deal with binary packed data
import sys
import struct
import builtins
import binascii
from collections import OrderedDict

__version__ = '1.15.35'

ENDIANNAMES = {
	'=': sys.byteorder,
	'@': sys.byteorder,
	'>': 'big',
	'!': 'big',
	'<': 'little'
}

SUBS = {}


def lreplace(s, reco, rep):
	if reco in s:
		l = list(s.partition(reco))
		l[1] = rep
		return ''.join(l)
	else:
		return s


def bin(val):
	if isinstance(val, int):
		return builtins.bin(val).lstrip('0b')
	elif type(val) in (bytes, bytearray, list, tuple):
		return ''.join([builtins.bin(b).lstrip('0b').zfill(8) for b in val])
	else:
		raise TypeError('Int, bytes or bytearray object is needed')


def hex(val):
	if isinstance(val, int):
		return builtins.hex(val).lstrip('0x')
	else:
		return binascii.hexlify(bytes(val)).decode('ascii')


def register_sub(sub, rep):
	SUBS[sub] = rep


class TypeUser (object):
	def __init__(self, byteorder='@'):
		self.byteorder = byteorder
	
	def pack(self, stct, *data):
		if stct[0] not in ENDIANNAMES.keys():
			stct = self.byteorder + stct
		return pack(stct, *data)
	
	def unpack(self, stct, data, refdata=()):
		if stct[0] not in ENDIANNAMES.keys():
			stct = self.byteorder + stct
		return unpack(stct, data, refdata)
	
	def unpack_from(self, stct, data, offset=0, refdata=(), getptr=False):
		if stct[0] not in ENDIANNAMES.keys():
			stct = self.byteorder + stct
		return unpack_from(stct, data, offset, refdata, getptr)


class TypeReader (TypeUser):
	def bit(self, n, bit, length=1):
		mask = ((2 ** length) - 1) << bit
		return (n & mask) >> (bit - length)
	
	def nibbles(self, n):
		return (n >> 4, n & 0xf)
	
	def signed_nibbles(self, n):
		high = (n >> 4)
		if high >= 8:
			high -= 16
		low = (n & 0xf)
		if low >= 8:
			low -= 16
		return high, low
	
	def uint8(self, data, ptr=0):
		return struct.unpack_from('%sB' % self.byteorder, data, ptr)[0], ptr + 1
	
	def uint16(self, data, ptr=0):
		return struct.unpack_from('%sH' % self.byteorder, data, ptr)[0], ptr + 2
	
	def uint24(self, data, ptr=0):
		return unpack_from('%sU' % self.byteorder, data, ptr)[0], ptr + 3
	
	def uint32(self, data, ptr=0):
		return struct.unpack_from('%sI' % self.byteorder, data, ptr)[0], ptr + 4
		
	def uint64(self, data, ptr=0):
		return struct.unpack_from('%sQ' % self.byteorder, data, ptr)[0], ptr + 8
	
	def int8(self, data, ptr=0):
		return struct.unpack_from('%sb' % self.byteorder, data, ptr)[0], ptr + 1
	
	def int16(self, data, ptr=0):
		return struct.unpack_from('%sh' % self.byteorder, data, ptr)[0], ptr + 2
	
	def int24(self, data, ptr=0):
		return unpack_from('%su' % self.byteorder, data, ptr)[0], ptr + 3
	
	def int32(self, data, ptr=0):
		return struct.unpack_from('%si' % self.byteorder, data, ptr)[0], ptr + 4
	
	def int64(self, data, ptr=0):
		return struct.unpack_from('%sq' % self.byteorder, data, ptr)[0], ptr + 8
	
	def float32(self, data, ptr=0):
		return struct.unpack_from('%sf' % self.byteorder, data, ptr)[0], ptr + 4
	
	def string(self, data, ptr=0):
		subdata = data[ptr:]
		try:
			end = subdata.index(0)
		except:
			end = -1
		if end == -1:
			return subdata.decode('utf-8'), ptr + len(subdata)
		else:
			return subdata[:end].decode('utf-8'), ptr + end + 1
	
	def utf16string(self, data, ptr):
		subdata = data[ptr:]
		try:
			end = subdata.index(b'\x00\x00')
			end += (end % 2)
		except:
			end = -1
		endian = 'le' if self.byteorder == '<' else 'be'
		if end == -1:
			return subdata.decode('utf-16-%s' % endian), ptr + len(subdata)
		else:
			return subdata[:end].decode('utf-16-%s' % endian), ptr + end + 2
	
	def color(self, data, ptr, format):
		format = format.upper().strip()
		if format == 'RGBA8':
			sz = 4
		elif format == 'RGB8':
			sz = 3
		if format in ('RGBA8', 'RGB8'):
			r = data[offset]
			g = data[offset + 1]
			b = data[offset + 2]
			if format == 'RGBA8':
				a = data[offset + 3]
			final = OrderedDict()
			final['RED'] = r
			final['GREEN'] = g
			final['BLUE'] = b
			if format == 'RGBA8':
				final['ALPHA'] = a
		return final, offset + sz


class TypeWriter (TypeUser):
	def nibbles(self, high, low):
		return (high << 4) + (low & 0xf)
	
	def signed_nibbles(self, high, low):
		if high < 0:
			high += 16
		if low < 0:
			low += 16
		return (high << 4) + (low & 0xf)
		
	def uint8(self, data):
		return struct.pack('%sB' % self.byteorder, data)
	
	def uint16(self, data):
		return struct.pack('%sH' % self.byteorder, data)
	
	def uint24(self, data):
		return pack('%sU' % self.byteorder, data)
	
	def uint32(self, data):
		return struct.pack('%sI' % self.byteorder, data)
	
	def uint64(self, data):
		return struct.pack('%sQ' % self.byteorder, data)
	
	def int8(self, data):
		return struct.pack('%sb' % self.byteorder, data)
	
	def int16(self, data):
		return struct.pack('%sh' % self.byteorder, data)
	
	def int24(self, data):
		return pack('%su' % self.byteorder, data)
	
	def int32(self, data):
		return struct.pack('%si' % self.byteorder, data)
	
	def int64(self, data):
		return struct.pack('%sq' % self.byteorder, data)
	
	def float32(self, data):
		return struct.pack('%sf' % self.byteorder, data)
	
	def string(self, data, align=0):
		s = data.encode('utf-8')
		if align < len(s) + 1:
			align = len(s) + 1
		return struct.pack('%s%ds' % (self.byteorder, align), s)
	
	def utf16string(self, data, align=0):
		endian = 'le' if self.byteorder == '<' else 'be'
		s = data.encode('utf-16-%s' % endian) + b'\x00\x00'
		if align < len(s) + 2:
			align = len(s) + 2
		return struct.pack('%s%ds' % (self.byteorder, align), s)
	
	def pad(self, num):
		return b'\x00' * num
	
	def align(self, data, alignment):
		padding = alignment - (len(data) % alignment)
		return b'\x00' * padding
	
	def color(self, data, format):
		format = format.upper()
		out = b''
		if format in ('RGB8', 'RGBA8'):
			out += self.uint8(data['RED'])
			out += self.uint8(data['GREEN'])
			out += self.uint8(data['BLUE'])
			if format == 'RGBA8':
				out += self.uint8(data['ALPHA'])
		return out


class FileReader (object):
	def __init__(self, file, byteorder='@'):
		self.file = file
		self.read = self.file.read
		self.write = self.file.write
		self.seek = self.file.seek
		self.tell = self.file.tell
		bs = self.file.tell()
		self.file.seek(0, 2)
		self.filelen = self.file.tell()
		self.file.seek(bs)
		self.r = TypeReader(byteorder)

	def uint8(self):
		return self.r.uint8(self.file.read(1), 0)[0]
	
	def uint16(self):
		return self.r.uint16(self.file.read(2), 0)[0]
	
	def uint24(self):
		return self.r.uint24(self.file.read(3), 0)[0]
	
	def uint32(self):
		return self.r.uint32(self.file.read(4), 0)[0]
	
	def uint64(self):
		return self.r.uint64(self.file.read(8), 0)[0]
	
	def int8(self):
		return self.r.int8(self.file.read(1), 8)[0]
	
	def int16(self):
		return self.r.int16(self.file.read(2), 8)[0]
	
	def int24(self):
		return self.r.int24(self.file.read(3), 0)[0]
	
	def int32(self):
		return self.r.int32(self.file.read(4), 0)[0]
	
	def int64(self):
		return self.r.int64(self.file.read(8), 0)[0]
	
	def float32(self):
		return self.r.float32(self.file.read(4), 0)[0]
	
	def string(self):
		c = 256
		s = b''
		while c not in (b'\x00', b''):
			c = self.file.read(1)
			if c != b'\x00':
				s += c
		return s
	
	def utf16string(self):
		c = 256
		s = ''
		while c not in (b'\x00\x00', b''):
			c = self.file.read(2)
			if c != b'\x00\x00':
				s += c.decode('utf-16-le' if self.r.byteorder == '<' else 'utf-16-be')
		return s


def _calcsize(stct, data):
	stct = stct.replace('u', 'hb').replace('U', 'HB')
	stct = stct.replace('(', '').replace(')', '')
	return struct.calcsize(stct)


def _pack(stct, data, byteorder):
	data = list(data)
	for i in range(len(data)):
		if isinstance(data[i], str):
			data[i] = data[i].encode('utf-8')
	finalstct = ''
	j = 0
	while j < len(stct):
		c = stct[j]
		j += 1
		if c in ('/', '#', '-'):
			idx = ''
			while stct[j].isdigit():
				idx += stct[j]
				j += 1
			idx = int(idx)
			if c == '/':
				res = str(data[idx])
			elif c == '#':
				res = str(data[idx])
			elif c == '-':
				res = str(len(data[idx]))
			finalstct += res
		else:
			finalstct += c
	stct = finalstct
	i = 0
	final = b''
	dataptr = 0
	while i < len(stct):
		c = stct[i]
		i += 1
		if c.isdigit():
			while stct[i].isdigit():
				c += stct[i]
				i += 1
			c = int(c)
			tp = stct[i]
			i += 1
			if tp.isalpha():
				if tp not in 'xas':
					s = tp * c
					for el in s:
						if el == 'n':
							final += data[dataptr] + b'\x00'
							dataptr += 1
						elif el == 'u':
							n = data[dataptr]
							dataptr += 1
							if abs(n) >= 0x800000:
								raise struct.error('Number for int24 > 8388607 (0x800000)')
							if n < 0:
								n = 0x1000000 + n
							final += n.to_bytes(3, ('little' if byteorder == '<' else 'big'))
						elif el == 'U':
							n = data[dataptr]
							dataptr += 1
							final += n.to_bytes(3, ('little' if byteorder == '<' else 'big'))
						else:
							final += struct.pack(byteorder + el, data[dataptr])
							dataptr += 1
				elif tp == 'a':
					pad = c - (len(final) % c)
					final += b'\x00' * pad
				elif tp == 'x':
					final += struct.pack(byteorder + str(c) + 's', binascii.unhexlify(data[dataptr].encode('ascii')))
					dataptr += 1
				else:
					final += struct.pack(byteorder + str(c) + tp, data[dataptr])
					dataptr += 1
			elif tp == '[':
				bracklvl = 1
				while bracklvl != 0:
					el = stct[i]
					i += 1
					if el == '[':
						bracklvl += 1
					elif el == ']':
						bracklvl -= 1
					tp += el
				datalist = data[dataptr]
				dataptr += 1
				for j in range(0, c):
					final += _pack(tp[1:-1], datalist[j], byteorder)
		elif c.isalpha():
			if c == 'n':
				final += data[dataptr] + b'\x00'
				dataptr += 1
			elif c == 'u':
				n = data[dataptr]
				dataptr += 1
				if abs(n) >= 0x800000:
					raise struct.error('Number for int24 > 8388607')
				if n < 0:
					n = 0x1000000 + n
				final += n.to_bytes(3, ENDIANNAMES[byteorder])
			elif c == 'U':
				n = data[dataptr]
				dataptr += 1
				final += n.to_bytes(3, ENDIANNAMES[byteorder])
			else:
				final += struct.pack(byteorder + c, data[dataptr])
				dataptr += 1
		elif c == '$':
			final += data[dataptr]
			return final
		elif c == '(':
			bracklvl = 1
			while bracklvl != 0:
				el = stct[i]
				i += 1
				if el == '(':
					bracklvl += 1
				elif el == ')':
					bracklvl -= 1
				c += el
			c = c[1:-1]
			datalist = data[dataptr]
			dataptr += 1
			final += _pack(c, datalist, byteorder)
		elif c == '{':
			bracklvl = 1
			while bracklvl > 0:
				el = stct[i]
				i += 1
				if el == '{':
					bracklvl += 1
				elif el == '}':
					bracklvl -= 1
				c += el
			c = c[1:-1]
			for datalist in data[dataptr]:
				final += _pack(c, datalist, byteorder)
			dataptr += 1
	return final


def pack(stct, *data):
	byteorder = stct[0] if stct[0] in '@=><!' else '@'
	stct = stct.lstrip('<>=!@')
	stct = stct.replace(' ', '')
	if len(SUBS) > 0:
		for sub in SUBS.keys():
			stct = stct.replace(sub, SUBS[sub])
	data = list(data)
	for i, el in enumerate(data):
		if isinstance(el, str):
			data[i] = el.encode('utf-8')
	packed = _pack(stct, data, byteorder)
	return packed


def _unpack(stct, data, byteorder, refdata=(), retused=False):
	lastidx = -1
	i = 0
	final = []
	ptr = 0
	while i < len(stct):
		indic = ''
		c = stct[i]
		i += 1
		if c.isdigit():
			while stct[i].isdigit():
				c += stct[i]
				i += 1
			c = int(c)
			tp = stct[i]
			i += 1
			if tp.isalpha():
				if tp == 'n':
					for i in range(0, c):
						null = data[ptr:].index(b'\x00')
						string = data[ptr:null + ptr]
						final.append(string)
						ptr += len(string) + 1
				elif c == 'u':
					for i in range(0, c):
						bnum = data[ptr:ptr + 3]
						ptr += 3
						endian = ENDIANNAMES[byteorder]
						num = int.from_bytes(bnum, endian)
						if num >= 0x800000:
							num = num - 0x1000000
						final.append(num)
				elif c == 'U':
					for i in range(0, c):
						bnum = data[ptr:ptr + 3]
						ptr += 3
						endian = ENDIANNAMES[byteorder]
						num = int.from_bytes(bnum, endian)
						final.append(num)
				elif tp == 'a':
					if (ptr % c) != 0:
						pad = c - (ptr % c)
					else:
						pad = 0
					final.append(data[ptr:ptr + pad])
					ptr += pad
				elif tp == 's':
					final.append(struct.unpack_from(byteorder + str(c) + tp, data, ptr)[0])
					ptr += c
				elif tp == 'x':
					final.append(binascii.hexlify(struct.unpack_from(byteorder + str(c) + 's', data, ptr)[0]).decode('ascii'))
					ptr += c
				else:
					c = tp * c
					for el in c:
						final += struct.unpack_from(byteorder + el, data, ptr)
						ptr += _calcsize(byteorder + el, data)
			elif tp == '[':
				bracklvl = 1
				while bracklvl != 0:
					el = stct[i]
					i += 1
					if el == '[':
						bracklvl += 1
					elif el == ']':
						bracklvl -= 1
					tp += el
				tp = tp[1:-1]
				ls = []
				for j in range(0, c):
					res, used = _unpack(tp, data[ptr:], byteorder)
					ls.append(res)
					ptr += used
				final.append(ls)
		elif c.isalpha():
			if c == 'n':
				null = data[ptr:].index(b'\x00')
				string = data[ptr:null + ptr]
				final.append(string)
				ptr += len(string) + 1
			elif c == 'u':
				bnum = data[ptr:ptr + 3]
				ptr += 3
				endian = ENDIANNAMES[byteorder]
				num = int.from_bytes(bnum, endian)
				if num >= 0x800000:
					num = num - 0x1000000
				final.append(num)
			elif c == 'U':
				bnum = data[ptr:ptr + 3]
				ptr += 3
				endian = ENDIANNAMES[byteorder]
				num = int.from_bytes(bnum, endian)
				final.append(num)
			else:
				final += struct.unpack_from(byteorder + c, data, ptr)
				ptr += _calcsize(byteorder + c, data)
		elif c == '$':
			final.append(data[ptr:])
			ptr = len(data)
			return final, ptr
		elif c == '(':
			bracklvl = 1
			while bracklvl != 0:
				el = stct[i]
				i += 1
				if el == '(':
					bracklvl += 1
				elif el == ')':
					bracklvl -= 1
				c += el
			c = c[1:-1]
			final.append(_unpack(c, data[ptr:ptr + _calcsize(byteorder + c, data)], byteorder)[0])
			ptr += _calcsize(byteorder + c, data)
		elif c == '{':
			bracklvl = 1
			while bracklvl != 0:
				el = stct[i]
				i += 1
				if el == '{':
					bracklvl += 1
				elif el == '}':
					bracklvl -= 1
				c += el
			ls = []
			c = c[1:-1]
			while ptr < len(data):
				res, used = _unpack(c, data[ptr:], byteorder)
				ls.append(res)
				ptr += used
			final.append(ls)
		elif c in ('/', '#', '-'):
			idx = ''
			j = i
			if stct[j].isalpha():
				indic = stct[j]
				j += 1
			while stct[j].isdigit():
				idx += stct[j]
				j += 1
			idx = int(idx)
			if c == '/':
				if indic == 'p':
					idx = -idx
				elif indic == 'l':
					idx += lastidx
				lastidx = idx
				if indic == 'p':
					lastidx = len(final) - idx
				res = str(final[idx])
				reco = '/' + indic + str(abs(idx))
				stct = lreplace(stct, reco, res)
				i -= 1
			elif c == '#':
				res = str(refdata[idx])
				reco = '#' + indic + str(idx)
				i -= 1
				stct = lreplace(stct, reco, res)
			elif c == '-':
				res = str(len(final[idx]))
				reco = '-' + indic + str(idx)
				stct = lreplace(stct, reco, res)
				i -= 1
	return final, ptr


def unpack(stct, data, refdata=()):
	byteorder = stct[0] if stct[0] in '@=><!' else '@'
	stct = stct.lstrip('<>=!@')
	stct = stct.replace(' ', '')
	if len(SUBS) > 0:
		for sub in SUBS.keys():
			stct = stct.replace(sub, SUBS[sub])
	unpacked, ptr = _unpack(stct, data, byteorder, refdata)
	return unpacked


def unpack_from(stct, data, offset=0, refdata=(), getptr=False):
	data = data[offset:]
	byteorder = stct[0] if stct[0] in '@=><!' else '@'
	stct = stct.lstrip('<>=!@')
	stct = stct.replace(' ', '')
	if len(SUBS) > 0:
		for sub in SUBS.keys():
			stct = stct.replace(sub, SUBS[sub])
	unpacked, ptr = _unpack(stct, data, byteorder, refdata)
	if getptr:
		return unpacked, ptr + offset
	else:
		return unpacked
