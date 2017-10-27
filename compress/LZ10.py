# -*- coding:utf-8 -*-
from util import error
from util.funcops import ClsFunc
import util.rawutil as rawutil


class decompressLZ10 (ClsFunc, rawutil.TypeReader):
	def main(self, content):
		self.byteorder = '<'
		self.readhdr(content)
		return self.decompress()
	
	def readhdr(self, content):
		if content[0] != 0x10:
			error('Invalid magic 0x%02x, expected 0x10' % content[0], 301)
		hdr = content[1:4] + b'\x00'
		self.data = content[4:]
		self.decsize = self.unpack('I', hdr)[0]
	
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
					infobs = rawutil.unpack_from('>H', self.data, ptr)[0]
					ptr += 2
					count = (infobs >> 12) + 3
					disp = (infobs & 0xfff) + 1
					for i in range(0, count):
						final.append(final[-disp])
				if len(final) >= self.decsize:
					break
		ret = bytes(final)
		return ret
