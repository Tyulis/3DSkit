# -*- coding:utf-8 -*-
# rawutil.py
# A single-file pure-python module to deal with binary packed data
import sys
import struct
import builtins
import binascii

__version__ = '2.2.4'

ENDIANNAMES = {
	'=': sys.byteorder,
	'@': sys.byteorder,
	'>': 'big',
	'!': 'big',
	'<': 'little'
}

SUBS = {}


def bin(val, align=0):
	if isinstance(val, int):
		return builtins.bin(val).lstrip('0b').zfill(align)
	elif type(val) in (bytes, bytearray, list, tuple):
		return ''.join([builtins.bin(b).lstrip('0b').zfill(8) for b in val]).zfill(align)
	else:
		raise TypeError('Int, bytes or bytearray object is needed')


def hex(val, align=0):
	if isinstance(val, int):
		return builtins.hex(val).lstrip('0x').zfill(align)
	else:
		return binascii.hexlify(bytes(val)).decode('ascii').zfill(align)


def hextoint(hx):
	return int(hx, 16)


def hextobytes(hx):
	if type(hx) == str:
		hx = hx.encode('ascii')
	return binascii.unhexlify(hx)


def register_sub(sub, rep):
	SUBS[sub] = rep


class _ClsFunc (object):
	def __new__(cls, *args, **kwargs):
		ins = object.__new__(cls)
		return ins.main(*args, **kwargs)


class TypeUser (object):
	def __init__(self, byteorder='@'):
		self.byteorder = byteorder
	
	def pack(self, stct, *data, out=None):
		byteorder = stct[0] if stct[0] in '@=><!' else self.byteorder
		stct = stct.lstrip('<>=!@')
		stct = stct.replace(' ', '')
		if len(SUBS) > 0:
			for sub in SUBS.keys():
				stct = stct.replace(sub, SUBS[sub])
		data = list(data)
		if hasattr(data[-1], 'write'):
			out = data.pop(-1)
		else:
			out = None
		packed = _pack(stct, data, byteorder, out)
		return packed
	
	def unpack(self, stct, data, refdata=()):
		byteorder = stct[0] if stct[0] in '@=><!' else self.byteorder
		stct = stct.lstrip('<>=!@')
		stct = stct.replace(' ', '')
		if len(SUBS) > 0:
			for sub in SUBS.keys():
				stct = stct.replace(sub, SUBS[sub])
		unpacked, ptr = _unpack(stct, data, 0, byteorder, refdata)
		return unpacked
	
	def unpack_from(self, stct, data, offset=0, refdata=(), getptr=False):
		byteorder = stct[0] if stct[0] in '@=><!' else self.byteorder
		stct = stct.lstrip('<>=!@')
		stct = stct.replace(' ', '')
		if len(SUBS) > 0:
			for sub in SUBS.keys():
				stct = stct.replace(sub, SUBS[sub])
		unpacked, ptr = _unpack(stct, data, offset, byteorder, refdata)
		if getptr:
			return unpacked, ptr
		else:
			return unpacked


class TypeReader (TypeUser):
	def tobits(self, n, align=8):
		return [int(bit) for bit in bin(n, align)]
		
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
		s = []
		zeroes = 0
		for i, c in enumerate(subdata):
			if c == 0:
				zeroes += 1
			else:
				zeroes = 0
			s.append(c)
			if zeroes >= 2 and i % 2 == 1:
				break
		endian = 'le' if self.byteorder == '<' else 'be'
		return bytes(s[:-2]).decode('utf-16-%s' % endian), ptr + i


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
		padding = alignment - (len(data) % alignment or alignment)
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


class _InternRef (object):
	def __init__(self, type='s', index=1):
		self.type, self.index = type, index
	
	def __repr__(self):
		return '/%s%d' % (self.type, self.index)


class _Sub (object):
	def __init__(self, type, structure):
		self.type, self.stct = type, structure
	
	def __repr__(self):
		return '%s%s%s' % (self.type[0], self.stct, self.type[1])


class _StructParser (object):
	def parse_struct(self, stct, refdata=()):
		ptr = 0
		final = []
		for i, el in enumerate(refdata):
			stct = stct.replace('#%d' % i, str(el))
		while ptr < len(stct):
			el = stct[ptr]
			ptr += 1
			if el.isdigit():
				countstr = ''
				while el.isdigit():
					countstr += el
					el = stct[ptr]
					ptr += 1
				count = int(countstr)
			elif el == '/':
				el = stct[ptr]
				ptr += 1
				indexstr = ''
				if el in 'sp':
					type = el
					el = stct[ptr]
					ptr += 1
				else:
					type = 's'
				while el.isdigit():
					indexstr += el
					el = stct[ptr]
					ptr += 1
				index = int(indexstr)
				count = _InternRef(type, index)
			else:
				count = 1
			if el in '([{':
				level = 1
				open = el
				close = ')' if el == '(' else (']' if el == '[' else '}')
				substruct = ''
				while level > 0:
					el = stct[ptr]
					ptr += 1
					if el == open:
						level += 1
					elif el == close:
						level -= 1
					substruct += el
				substruct = substruct[0:-1]  #removes the last close bracket
				el = _Sub(open + close, self.parse_struct(substruct))
			token = (count, el)
			final.append(token)
		return final


def _calcsize(stct, data):
	stct = stct.replace('u', 'hb').replace('U', 'HB')
	stct = stct.replace('(', '').replace(')', '')
	return struct.calcsize(stct)


class _unpack (_ClsFunc, _StructParser):
	def main(self, stct, data, ptr, byteorder, refdata=()):
		self.data = data
		self.byteorder = byteorder
		self.endianname = ENDIANNAMES[byteorder]
		stct = self.parse_struct(stct, refdata)
		if hasattr(self.data, 'read'):
			self.data.seek(ptr)
			return self.unpack_file(stct)
		else:
			self.offset = ptr
			self.data = self.data[ptr:]
			self.ptr = 0
			return self.unpack(stct)
	
	def unpack(self, stct):
		groupbase = self.ptr
		final = []
		for token in stct:
			count, el = token
			if isinstance(count, _InternRef):
				if count.type == 's':
					count = final[count.index]
				elif count.type == 'p':
					count = final[-count.index]
			if isinstance(el, _Sub):
				if el.type == '()':
					final.append(self.unpack(el.stct)[0])
				elif el.type == '[]':
					final.append([self.unpack(el.stct)[0] for i in range(count)])
				elif el.type == '{}':
					sub = []
					while self.ptr < len(self.data):
						sub.append(self.unpack(el.stct)[0])
					final.append(sub)
					break
			else:
				if el in 'uU':
					for _ in range(count):
						sub = self.data[self.ptr: self.ptr + 3]
						self.ptr += 3
						num = int.from_bytes(sub, self.endianname)
						if num > 0x7fffff and el == 'u':
							num -= 0x1000000
						final.append(num)
				elif el == 'x':
					self.ptr += count
				elif el == 'X':
					sub = self.data[self.ptr: self.ptr + count]
					self.ptr += count
					final.append(hex(sub))
				elif el == 'a':
					subptr = self.ptr - groupbase
					padding = count - (subptr % count or count)
					self.ptr += padding
				elif el == 'n':
					for _ in range(count):
						try:
							null = self.data[self.ptr:].index(b'\x00')
							s = self.data[self.ptr: self.ptr + null]
						except ValueError:
							s = self.data[self.ptr:]
						self.ptr += len(s) + 1  #skips the null byte
						final.append(s)
				elif el == 's':
					final.append(self.data[self.ptr: self.ptr + count])
					self.ptr += count
				elif el == '?':
					for _ in range(count):
						final.append(bool(self.data[self.ptr]))
						self.ptr += 1
				elif el == '$':
					final.append(self.data[self.ptr:])
					break
				else:
					#Avoids copy of the entire data
					substruct = '%s%d%s' % (self.byteorder, count, el)
					length = struct.calcsize(substruct)
					subdata = self.data[self.ptr: self.ptr + length]
					self.ptr += length
					final += struct.unpack(substruct, subdata)
		return final, self.ptr + self.offset
	
	def unpack_file(self, stct):
		groupbase = self.data.tell()
		final = []
		for token in stct:
			count, el = token
			if isinstance(count, _InternRef):
				if count.type == 's':
					count = final[count.index]
				elif count.type == 'p':
					count = final[-count.index]
			if isinstance(el, _Sub):
				if el.type == '()':
					final.append(self.unpack_file(el.stct)[0])
				elif el.type == '[]':
					final.append([self.unpack_file(el.stct)[0] for i in range(count)])
				elif el.type == '{}':
					sub = []
					while self.ptr < len(self.data):
						sub.append(self.unpack_file(el.stct)[0])
					final.append(sub)
					break
			else:
				if el in 'uU':
					for _ in range(count):
						sub = self.data.read(3)
						num = int.from_bytes(sub, self.endianname)
						if num > 0x7fffff and el == 'u':
							num -= 0x1000000
						final.append(num)
				elif el == 'x':
					self.data.seek(count, 1)
				elif el == 'X':
					sub = self.data.read(count)
					final.append(hex(sub))
				elif el == 'a':
					subptr = self.data.tell() - groupbase
					padding = count - (subptr % count or count)
					self.data.seek(padding, 1)
				elif el == 'n':
					for _ in range(count):
						s = b''
						while not s.endswith(b'\x00'):
							byte = self.data.read(1)
							if byte == b'':
								break
							s += byte
						final.append(s.rstrip(b'\x00'))
				elif el == 's':
					final.append(self.data.read(count))
				elif el == '?':
					for _ in range(count):
						final.append(bool(self.data.read(1)))
				elif el == '$':
					final.append(self.data.read())
					break
				else:
					#Avoids copy of the entire data
					substruct = '%s%d%s' % (self.byteorder, count, el)
					length = struct.calcsize(substruct)
					subdata = self.data.read(length)
					final += struct.unpack(substruct, subdata)
		return final, self.data.tell()


class _pack (_StructParser, _ClsFunc):
	def main(self, stct, data, byteorder, out):
		self.byteorder = byteorder
		self.endianname = ENDIANNAMES[byteorder]
		stct = self.parse_struct(stct)
		if out is None:
			self.final = b''
			self.pack(stct, data)
			return self.final
		else:
			self.final = out
			self.pack_file(stct, data)
	
	def pack(self, stct, data):
		ptr = 0
		groupbase = len(self.final)
		for token in stct:
			count, el = token
			if isinstance(count, _InternRef):
				if count.type == 's':
					count = data[count.index]
				elif count.type == 'p':
					count = data[ptr - count.index]
			if isinstance(el, _Sub):
				if el.type == '()':
					self.pack(el.stct, data[ptr])
					ptr += 1
				elif el.type == '[]':
					[self.pack(el.stct, data[ptr][i]) for i in range(count)]
					ptr += 1
				elif el.type == '{}':
					for subdata in data[ptr]:
						self.pack(el.stct, subdata)
					ptr += 1
			else:
				if el in 'Uu':
					for _ in range(count):
						num = data[ptr]
						ptr += 1
						if num < 0 and el == 'u':
							num += 0x1000000
						self.final += num.to_bytes(3, self.endianname)
				elif el == 'x':
					self.final += b'\x00' * count
				elif el == 'X':
					sub = hextobytes(data[ptr])
					if len(sub) != count:
						raise struct.error('Given string of %d bytes, expected %d' % (len(sub), count))
					ptr += 1
					self.final += sub
				elif el == 's':
					sub = data[ptr]
					if type(sub) == str:
						sub = sub.encode('utf-8')
					if len(sub) != count:
						raise struct.error('Given string of %d bytes, expected %d' % (len(sub), count))
					ptr += 1
					self.final += sub
				elif el == 'n':
					for _ in range(count):
						sub = data[ptr]
						ptr += 1
						if type(sub) == str:
							sub = sub.encode('utf-8')
						self.final += sub + b'\x00'
				elif el == 'a':
					subptr = len(self.final) - groupbase
					length = count - (subptr % count or count)
					self.final += length * b'\x00'
				elif el == '?':
					for _ in range(count):
						self.final += struct.pack('B', data[ptr])
						ptr += 1
				elif el == '$':
					self.final += data[ptr]
					break
				else:
					substruct = '%s%d%s' % (self.byteorder, count, el)
					subdata = data[ptr: ptr + count]
					ptr += count
					self.final += struct.pack(substruct, *subdata)
	
	def pack_file(self, stct, data):
		ptr = 0
		groupbase = self.final.tell()
		for token in stct:
			count, el = token
			if isinstance(count, _InternRef):
				if count.type == 's':
					count = data[count.index]
				elif count.type == 'p':
					count = data[ptr - count.index]
			if isinstance(el, _Sub):
				if el.type == '()':
					self.pack_file(el.stct, data[ptr])
					ptr += 1
				elif el.type == '[]':
					[self.pack_file(el.stct, data[ptr][i]) for i in range(count)]
					ptr += 1
				elif el.type == '{}':
					for subdata in data[ptr]:
						self.pack_file(el.stct, subdata)
					ptr += 1
			else:
				if el in 'Uu':
					for _ in range(count):
						num = data[ptr]
						ptr += 1
						if num < 0 and el == 'u':
							num += 0x1000000
						self.final.write(num.to_bytes(3, self.endianname))
				elif el == 'x':
					self.final.write(b'\x00' * count)
				elif el == 'X':
					sub = hextobytes(data[ptr])
					if len(sub) != count:
						raise struct.error('Given string of %d bytes, expected %d' % (len(sub), count))
					ptr += 1
					self.final.write(sub)
				elif el == 's':
					sub = data[ptr]
					if type(sub) == str:
						sub = sub.encode('utf-8')
					if len(sub) != count:
						raise struct.error('Given string of %d bytes, expected %d' % (len(sub), count))
					ptr += 1
					self.final.write(sub)
				elif el == 'n':
					for _ in range(count):
						sub = data[ptr]
						ptr += 1
						if type(sub) == str:
							sub = sub.encode('utf-8')
						self.final.write(sub + b'\x00')
				elif el == 'a':
					subptr = self.final.tell() - groupbase
					length = count - (subptr % count or count)
					self.final.write(length * b'\x00')
				elif el == '?':
					for _ in range(count):
						self.final.write(struct.pack('B', data[ptr]))
						ptr += 1
				elif el == '$':
					self.final.write(data[ptr])
					break
				else:
					substruct = '%s%d%s' % (self.byteorder, count, el)
					subdata = data[ptr: ptr + count]
					ptr += count
					self.final.write(struct.pack(substruct, *subdata))


def unpack(stct, data, refdata=()):
	byteorder = stct[0] if stct[0] in '@=><!' else '@'
	stct = stct.lstrip('<>=!@')
	stct = stct.replace(' ', '')
	if len(SUBS) > 0:
		for sub in SUBS.keys():
			stct = stct.replace(sub, SUBS[sub])
	unpacked, ptr = _unpack(stct, data, 0, byteorder, refdata)
	return unpacked


def unpack_from(stct, data, offset=0, refdata=(), getptr=False):
	byteorder = stct[0] if stct[0] in '@=><!' else '@'
	stct = stct.lstrip('<>=!@')
	stct = stct.replace(' ', '')
	if len(SUBS) > 0:
		for sub in SUBS.keys():
			stct = stct.replace(sub, SUBS[sub])
	unpacked, ptr = _unpack(stct, data, offset, byteorder, refdata)
	if getptr:
		return unpacked, ptr
	else:
		return unpacked


def pack(stct, *data):
	byteorder = stct[0] if stct[0] in '@=><!' else '@'
	stct = stct.lstrip('<>=!@')
	stct = stct.replace(' ', '')
	if len(SUBS) > 0:
		for sub in SUBS.keys():
			stct = stct.replace(sub, SUBS[sub])
	data = list(data)
	if hasattr(data[-1], 'write'):
		out = data.pop(-1)
	else:
		out = None
	packed = _pack(stct, data, byteorder, out)
	return packed

if __name__ == '__main__':
	#test
	s = '>4s2I/p1[H(2B)] n4a 5X2x? 2u'
	raw = b'TESTaaaa\x00\x00\x00\x02GGhiJJklRETEST\x00\x00YBOOM\x00\x00\x01333666'
	data = unpack(s, raw)
	f = pack(s, *data)
	assert f == raw
	file = open('test.bin', 'wb')
	pack(s, *data, file)
	file.close()
	file = open('test.bin', 'rb')
	d = unpack(s, file)
	assert d == data
