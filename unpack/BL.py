# -*- coding:utf-8 -*-
import util.rawutil as rawutil
from util.fileops import *
from util.funcops import ClsFunc, byterepr
from unpack._formats import get_ext

BL_TABLE_STRUCT = '2sH /1[I] 128a'


class extractBL (ClsFunc, rawutil.TypeReader):
	def main(self, filename, data, verbose, endian, opts={}):
		self.byteorder = endian
		self.verbose = verbose
		self.outdir = make_outdir(filename)
		offsets = self.read_table(data)
		files = self.extract_files(offsets, data)
		self.write_files(files)
	
	def read_table(self, data):
		tbl = self.unpack_from(BL_TABLE_STRUCT, data, 0)
		if tbl[0] != b'BL':
			error('Invalid magic %s, expected BL' % byterepr(tbl[0]), 301)
		filecount = tbl[1]
		offsets = [el[0] for el in tbl[2]]
		if self.verbose:
			print('File count: %d' % filecount)
		return offsets
	
	def extract_files(self, offsets, data):
		files = []
		for i in range(0, len(offsets) - 1):  #the last is total file length
			data.seek(offsets[i])
			length = offsets[i + 1] - offsets[i]
			files.append(data.read(length))
		return files
	
	def write_files(self, files):
		for i, filedata in enumerate(files):
			ext = get_ext(filedata)
			name = path(self.outdir, '%03d%s' % (i, ext))
			bwrite(filedata, name)
