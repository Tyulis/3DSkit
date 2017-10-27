# -*- coding:utf-8 -*-
from util.funcops import ClsFunc
import util.rawutil as rawutil


class compressLZ11 (ClsFunc, rawutil.TypeWriter):
	def main(self, content):
		self.byteorder = '<'
		hdr = self.makeheader(content)
		compressed = self.compress(content)
		final = hdr + compressed
		return final
	
	def makeheader(self, content):
		hdr = b'\x11'
		hdr += self.uint24(len(content))
		return hdr
	
	def compress(self, data):
		return b''
	

class decompressLZ11 (ClsFunc, rawutil.TypeReader):
	def main(self, content):
		self.byteorder = '<'
		self.readhdr(content)
		return self.decompress()
	
	def readhdr(self, content):
		if content[0] != 0x11:
			error('Invalid magic 0x%02x, expected 0x11' % content[0], 301)
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
					byte, ptr = self.uint8(self.data, ptr)
					indic = byte >> 4
					if indic == 0:
						count = byte << 4
						byte, ptr = self.uint8(self.data, ptr)
						count += byte >> 4
						count += 0x11
					elif indic == 1:
						byte2, ptr = self.uint8(self.data, ptr)
						count = ((byte & 0xf) << 12) + (byte2 << 4)
						byte, ptr = self.uint8(self.data, ptr)
						count += byte >> 4
						count += 0x111
					else:
						count = indic + 1
					dbyte, ptr = self.uint8(self.data, ptr)
					disp = ((byte & 0xf) << 8) + dbyte + 1
					for i in range(0, count):
						final.append(final[-disp])
				if len(final) >= self.decsize:
					break
		return bytes(final)
