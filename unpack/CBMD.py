# -*- coding:utf-8 -*-
import os
import compress
from util import error, ENDIANS
from util.funcops import ClsFunc
from util.fileops import make_outdir, bread, bwrite, makedirs
import util.rawutil as rawutil

CBMD_HEADER_STRUCT = '4s2I(13I)44sI'
OUT_NAMES = (
	'EUR-EN.cgfx',
	'EUR-FR.cgfx',
	'EUR-DE.cgfx',
	'EUR-IT.cgfx',
	'EUR-ES.cgfx',
	'EUR-DK.cgfx',
	'EUR-PT.cgfx',
	'EUR-RU.cgfx',
	'JPN-JP.cgfx',
	'USA-EN.cgfx',
	'USA-FR.cgfx',
	'USA-ES.cgfx',
	'USA-PT.cgfx'
)

class extractCBMD (ClsFunc, rawutil.TypeReader):
	def main(self, filename, data):
		self.byteorder = '<'  #exists only on 3ds
		self.outdir = make_outdir(filename)
		self.readheader(data)
		self.extract(data)
	
	def readheader(self, data):
		hdata = self.unpack_from(CBMD_HEADER_STRUCT, data, 0)
		if hdata[0] != b'CBMD':
			error('Invalid magic : %s' % hdata[0])
		self.commonoffset = hdata[2]
		self.regionoffsets = hdata[3]
		self.cwavoffset = hdata[5]
	
	def extract(self, data):
		offsets = {i: off for i, off in enumerate(self.regionoffsets) if off != 0}
		if len(offsets) == 0:
			filedata = data[self.commonoffset:]
		else:
			filedata = data[self.commonoffset:self.commonoffset + sorted(list(offsets.values()))[0]]
		filedata = compress.decompress(filedata, compress.recognize(filedata), self.byteorder)
		bwrite(filedata, self.outdir + 'common.cgfx')
		sortkeys = sorted(list(offsets.keys()))
		for i in offsets.keys():
			if i != max(sortkeys):
				start = offsets[i]
				end = sortkeys[sortkeys.index(i) + 1]
				filedata = data[start:end]
			else:
				filedata = data[offsets[i]:]
			filedata = compress.decompress(filedata, compress.recognize(filedata), self.byteorder)
			bwrite(filedata, OUT_NAMES[i])
