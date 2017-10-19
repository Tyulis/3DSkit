# -*- coding:utf-8 -*-
import util.rawutil as rawutil
from util.fileops import *

BL_TABLE_STRUCT = '2sH /1[I] 128a'


class extractBL (rawutil.TypeReader):
	def __init__(self, filename, data, endian, opts={}):
		self.byteorder = endian
		self.outdir = make_outdir(filename)
		offsets = self.read_table(data)
	
	def read_table(self, data):
		tbl = self.unpack_from(BL_TABLE_STRUCT, data, 0)
		if tbl[0] != b'BL':
			error('Invalid magic : %s' % tbl[0])
		filecount = tbl[1]
		offsets = [el[0] for el in tbl[2]]
		return offsets
