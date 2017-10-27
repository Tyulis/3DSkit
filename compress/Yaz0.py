# -*- coding:utf-8 -*-
from util import error
from util.funcops import ClsFunc, byterepr
import util.rawutil as rawutil

Yaz0_HEADER_STRUCT = '4sI(2I)'


class decompressYaz0 (ClsFunc, rawutil.TypeReader):
	def main(self, content):
		self.byteorder = '>'
		data = self.readheader(content)
		return self.decompress(data)
	
	def readheader(self, data):
		magic, self.unco_len, unknown = self.unpack_from(Yaz0_HEADER_STRUCT, data, 0)
		if magic != b'Yaz0':
			error('Invalid magic %s, expected Yaz0' % byterepr(magic))
		return data[0x10:]
	
	def decompress(self, data):
		final = []
		ptr = 0
		while len(final) < self.unco_len:
			hdr, ptr = self.uint8(data, ptr)
			hdr = rawutil.bin(hdr).zfill(8)
			for b in hdr:
				if b == '1':
					byte, ptr = self.uint8(data, ptr)
					final.append(byte)
				else:
					indic = self.uint8(data, ptr)[0]
					if indic >> 4 == 0:
						ref, ptr = self.uint24(data, ptr)
						disp = (ref >> 8) + 1
						size = (ref & 0xff) + 0x12
					else:
						ref, ptr = self.uint16(data, ptr)
						disp = (ref & 0xfff) + 1
						size = (ref >> 12) + 2
					for i in range(0, size):
						final.append(final[-disp])
				if ptr >= len(data):
					break
		return bytes(final)
