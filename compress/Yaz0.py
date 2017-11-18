# -*- coding:utf-8 -*-
from util import error
from util.funcops import ClsFunc, byterepr
import util.rawutil as rawutil

Yaz0_HEADER_STRUCT = '4sI(2I)'


class decompressYaz0 (ClsFunc, rawutil.TypeReader):
	def main(self, file, out, verbose):
		self.byteorder = '>'
		self.verbose = verbose
		self.file = file
		self.out = out
		self.file.seek(0, 2)
		self.inlen = self.file.tell()
		self.file.seek(0)
		self.readheader()
		self.decompress()
	
	def readheader(self):
		magic, self.unco_len, unknown = self.unpack_from(Yaz0_HEADER_STRUCT, self.file, 0)
		if magic != b'Yaz0':
			error('Invalid magic %s, expected Yaz0' % byterepr(magic))
	
	def decompress(self):
		self.out.write(b'\x00' * self.unco_len)
		self.out.seek(0)
		actlen = 0
		while actlen < self.unco_len:
			try:
				flags = ord(self.file.read(1))
			except TypeError:
				break
			mask = 1 << 7
			for i in range(8):
				if flags & mask:
					byte = self.file.read(1)
					self.out.write(byte)
					actlen += 1
				else:
					ref = (ord(self.file.read(1)) << 8) | ord(self.file.read(1))
					disp = (ref & 0xfff) + 1
					if ref >> 12:
						size = (ref >> 12) + 2
					else:
						size = ord(self.file.read(1)) + 0x12
					if size > disp:
						self.out.seek(-disp, 1)
						buffer = bytearray(self.out.read(disp + size))
						for j in range(size):
							buffer[disp + j] = buffer[j]
						self.out.seek(-(disp + size), 1)
						self.out.write(buffer)
					else:
						self.out.seek(-disp, 1)
						ref = self.out.read(size)
						self.out.seek(disp - size, 1)
						self.out.write(ref)
					actlen = self.out.tell()
				if self.file.tell() >= self.inlen:
					break
				mask >>= 1
