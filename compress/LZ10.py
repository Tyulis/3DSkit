# -*- coding:utf-8 -*-
from util import error
from util.funcops import ClsFunc
import util.rawutil as rawutil
from collections import defaultdict
from operator import itemgetter


class LZ10SlidingWindow (object):
	def __init__(self, data):
		self.hash = defaultdict(list)
		self.data = data
		self.full = False
		self.start = 0
		self.stop = 0
		self.index = 0
		#some constants
		self.size = 4096
		self.mindisp = 2
		self.startdisp = 1
		self.minmatch = 3
		self.maxmatch = 0xf + 3
	
	def next(self):
		if self.index < self.startdisp - 1:
			self.index += 1
			return
		if self.full:
			old = self.data[self.start]
			assert self.hash[old][0] == self.start
			self.hash[old].pop(0)
		itm = self.data[self.stop]
		self.hash[itm].append(self.stop)
		self.stop += 1
		self.index += 1
		if self.full:
			self.start += 1
		else:
			if self.size <= self.stop:
				self.full = True
	
	def skip(self, num):
		[self.next() for i in range(num)]
	
	def search(self):
		counts = []
		indices = self.hash[self.data[self.index]]
		for idx in indices:
			matchlen = self.match(idx, self.index)
			if matchlen >= self.minmatch:
				disp = self.index - idx
				if disp >= self.mindisp:
					counts.append((matchlen, -disp))
					if matchlen >= self.maxmatch:
						return counts[-1]
		if counts:
			return max(counts, key=itemgetter(0))
	
	def match(self, start, datastart):
		size = self.index - start
		if size == 0:
			return 0
		matchlen = 0
		it = range(min(len(self.data) - datastart, self.maxmatch))
		for i in it:
			if self.data[start + (i % size)] == self.data[datastart + i]:
				matchlen += 1
			else:
				break
		return matchlen


class compressLZ10 (ClsFunc, rawutil.TypeWriter):
	def main(self, content, verbose):
		self.byteorder = '>'
		self.verbose = verbose
		hdr = self.makeheader(content)
		compressed = self.compress(content)
		final = hdr + compressed
		return final
	
	def makeheader(self, content):
		hdr = b'\x10'
		hdr += self.pack('<U', len(content))
		return hdr
	
	def packflags(self, flags):
		n = 0
		for i, flag in enumerate(flags):
			n |= flag << (7 - i)
		return n
	
	def compress(self, data):
		length = 0
		final = b''
		for tokens in self.chunkgen(self._compress(data), 8):
			flags = [type(token) == tuple for token in tokens]
			final += self.uint8(self.packflags(flags))
			for token in tokens:
				if type(token) == tuple:
					count, disp = token
					count -= 3
					disp = -disp - 1
					assert 0 <= disp < 4096
					sh = (count << 12) | disp
					final += self.uint16(sh)
				else:
					final += self.uint8(token)
				length += 1
				length += sum([2 if flag else 1 for flag in flags])
		padding = 4 - (length % 4 or 4)
		final += b'\xff' * padding
		return final
	
	def chunkgen(self, it, n):
		buffer = []
		for x in it:
			buffer.append(x)
			if n <= len(buffer):
				yield buffer
				buffer = []
		if buffer:
			yield buffer
	
	def _compress(self, data):
		window = LZ10SlidingWindow(data)
		i = 0
		while True:
			if len(data) <= i:
				break
			match = window.search()
			if match:
				yield match
				window.skip(match[0])
				i += match[0]
			else:
				yield data[i]
				window.next()
				i += 1


class decompressLZ10 (ClsFunc, rawutil.TypeReader):
	def main(self, content, verbose):
		self.byteorder = '>'
		self.verbose = verbose
		self.readhdr(content)
		return self.decompress()
	
	def readhdr(self, content):
		if content[0] != 0x10:
			error('Invalid magic 0x%02x, expected 0x10' % content[0], 301)
		self.data = content[4:]
		self.decsize = self.unpack_from('<U', content, 1)[0]
	
	def decompress(self):
		ptr = 0
		final = []
		while len(final) < self.decsize:
			flags = self.tobits(self.data[ptr])
			ptr += 1
			for flag in flags:
				if flag == 0:
					byte, ptr = self.uint8(self.data, ptr)
					final.append(byte)
				else:
					infobs = self.unpack_from('H', self.data, ptr)[0]
					ptr += 2
					count = (infobs >> 12) + 3
					disp = (infobs & 0xfff) + 1
					for i in range(0, count):
						final.append(final[-disp])
				if len(final) >= self.decsize:
					break
		ret = bytes(final)
		return ret
