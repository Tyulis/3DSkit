# -*- coding:utf-8 -*-
import os
from util import error, ENDIANS
from util.funcops import byterepr
from util.fileops import *
import util.rawutil as rawutil

CGFX_HEADER_STRUCT = '4s 2H3I'
CGFX_HEADER_NAMES = 'magic, bom, headerlen, version, filelen, entrycount'
CGFX_DATA_HEADER_STRUCT = '4sI 16[2I]'
CGFX_DATA_HEADER_NAMES = 'magic, size, dictinfo'

class extractCGFX (rawutil.TypeReader):
	def __init__(self, filename, file, verbose, opts={}):
		self.outdir = make_outdir(filename)
		self.verbose = verbose
		self.file = file
		self.readheader()
		self.readDATA()
	
	def readheader(self):
		self.byteorder = ENDIANS[rawutil.unpack_from('>H', self.file, 4)[0]]
		hdata = self.unpack_from(CGFX_HEADER_STRUCT, self.file, 0, CGFX_HEADER_NAMES)
		if hdata.magic != b'CGFX':
			error('Invalid magic: %s' % byterepr(hdata.magic), error.invalidmagic)
		self.version = hdata.version
	
	def readDATA(self):
		header = self.unpack_from(CGFX_DATA_HEADER_STRUCT, self.file, None, CGFX_DATA_HEADER_NAMES)
		if header.magic != b'DATA':
			error('Invalid DATA magic: %s' % byterepr(header.magic), error.invalidmagic)

	def extract(self):
		#Code to really extract files
		pass
