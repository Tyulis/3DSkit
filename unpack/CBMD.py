# -*- coding:utf-8 -*-
import os
import compress
from io import BytesIO
from util import error
from util.utils import ClsFunc, byterepr
from util.filesystem import make_outdir, bread, bwrite
import util.rawutil as rawutil

CBMD_HEADER_STRUCT = '4s2I(13I) 44sI'
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
	def main(self, filename, data, verbose, opts={}):
		self.byteorder = '<'  #exists only on 3ds
		self.verbose = verbose
		self.outdir = make_outdir(filename)
		self.readheader(data)
		self.extract(data)
	
	def readheader(self, data):
		hdata = self.unpack_from(CBMD_HEADER_STRUCT, data, 0)
		if hdata[0] != b'CBMD':
			error.InvalidMagicError('Invalid magic %s, expected CBMD' % byterepr(hdata[0]))
		self.commonoffset = hdata[2]
		self.regionoffsets = hdata[3]
		self.cwavoffset = hdata[5]
	
	def extract(self, data):
		offsets = {i: off for i, off in enumerate(self.regionoffsets) if off != 0}
		data.seek(self.commonoffset)
		if len(offsets) == 0:
			indata = BytesIO(data.read())
		else:
			indata = BytesIO(data.read(sorted(list(offsets.values()))[0]))
		filedata = BytesIO()
		compress.decompress(indata, filedata, compress.recognize(indata), self.verbose)
		filedata.seek(0)
		bwrite(filedata.read(), self.outdir + 'common.cgfx')
		sortkeys = sorted(list(offsets.keys()))
		for i in offsets.keys():
			if i != max(sortkeys):
				start = offsets[i]
				end = sortkeys[sortkeys.index(i) + 1]
				data.seek(start)
				indata = BytesIO(data.read(end - start))
			else:
				data.seek(offsets[i])
				indata = BytesIO(data.read())
			filedata = BytesIO()
			compress.decompress(indata, filedata, compress.recognize(indata), self.verbose)
			filedata.seek(0)
			bwrite(filedata.read(), OUT_NAMES[i])
