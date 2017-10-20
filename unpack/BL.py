# -*- coding:utf-8 -*-
import util.rawutil as rawutil
from util.fileops import *
from util.funcops import ClsFunc
from unpack._formats import get_ext

BL_TABLE_STRUCT = '2sH /1[I] 128a'


class extractBL (ClsFunc, rawutil.TypeReader):
	def main(self, filename, data, endian, opts={}):
		self.byteorder = endian
		self.outdir = make_outdir(filename)
		offsets = self.read_table(data)
		files = self.extract_files(offsets, data)
		self.write_files(files)
	
	def read_table(self, data):
		tbl = self.unpack_from(BL_TABLE_STRUCT, data, 0)
		if tbl[0] != b'BL':
			error('Invalid magic : %s' % tbl[0])
		filecount = tbl[1]
		offsets = [el[0] for el in tbl[2]]
		return offsets
	
	def extract_files(self, offsets, data):
		files = []
		for i in range(0, len(offsets) - 1):  #the last is file length
			files.append(data[offsets[i]: offsets[i + 1]])
		return files
	
	def write_files(self, files):
		for i, filedata in enumerate(files):
			ext = get_ext(filedata)
			name = path(self.outdir, '%d%s' % (i, ext))
			bwrite(filedata, name)
