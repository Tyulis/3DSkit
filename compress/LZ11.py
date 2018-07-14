# -*- coding:utf-8 -*-
import numpy as np
from util import libkit
from util.utils import ClsFunc
import util.rawutil as rawutil


class compressLZ11 (ClsFunc, rawutil.TypeWriter):
	def main(self, file, out, verbose):
		self.byteorder = '>'
		self.verbose = verbose
		self.file = file
		self.out = out
		self.file.seek(0, 2)
		self.datalen = self.file.tell()
		self.file.seek(0)
		self.out.seek(0)
		self.makeheader()
		self.compress()
	
	def makeheader(self):
		self.out.write(b'\x11')
		self.pack('<U', self.datalen, self.out)
	
	def compress(self):
		data = np.ascontiguousarray(np.fromstring(self.file.read(), dtype=np.uint8))
		out = np.ascontiguousarray(np.zeros(len(data) * 2, dtype=np.uint8))
		outsize = libkit.compressLZ11(data, out, self.datalen)
		self.out.write(out[:outsize].tostring())
		ptr = self.out.tell()
		padding = 4 - (ptr % 4 or 4)
		self.out.write(b'\x00' * padding)

#inspirated from nlzss

class decompressLZ11 (ClsFunc, rawutil.TypeReader):
	def main(self, file, out, verbose):
		self.byteorder = '>'
		self.verbose = verbose
		self.file = file
		self.out = out
		self.file.seek(0)
		self.out.seek(0)
		self.readhdr()
		self.decompress()
	
	def readhdr(self):
		magic, self.decsize = self.unpack_from('<BU', self.file)
		if magic != 0x11:
			error.InvalidMagicError('Invalid magic 0x%02x, expected 0x11' % magic)
		if self.decsize == 0:
			raise RuntimeError('INTERNAL. SHOULD BE CAUGHT (Recognition error)')
	
	def decompress(self):
		data = np.fromstring(self.file.read(), dtype=np.uint8)
		out = np.zeros(self.decsize, np.uint8)
		libkit.decompressLZ11(data, out, len(data), self.decsize)
		self.out.write(out)