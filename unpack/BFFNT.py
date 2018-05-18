# -*- coding:utf-8 -*-
import os
import json
import numpy as np
from PIL import Image
from util import error, ENDIANS
from util.utils import byterepr, ClsFunc
from util.filesystem import *
import util.rawutil as rawutil

try:
	import c3DSkit
except:
	c3DSkit = None

CMAP_DIRECT = 0x00
CMAP_TABLE = 0x01
CMAP_SCAN = 0x02

FORMAT_NAMES_CTR = {
	0x00: 'RGBA8', 0x01: 'RGB8',
	0x02: 'RGBA5551', 0x03: 'RGB565',
	0x04: 'RGBA4', 0x05: 'LA8',
	0x06: 'HILO8', 0x07: 'L8',
	0x08: 'A8', 0x09: 'LA4',
	0x0A: 'L4', 0x0B: 'A4',
	0x0C: 'ETC1', 0x0D: 'ETC1A4',
}

FORMAT_NAMES_CAFE = {
	0x00: 'RGBA8', 0x01: 'RGB8',
	0x02: 'RGBA5551', 0x03: 'RGB565',
	0x04: 'RGBA4', 0x05: 'LA8',
	0x06: 'LA4', 0x07: 'A4',
	0x08: 'A8', 0x09: 'BC1',
	0x0A: 'BC2', 0x0B: 'BC3',
	0x0C: 'BC4', 0x0D: 'BC5',
	0x0E: 'RGBA8_SRGB', 0x0F: 'BC1_SRGB',
	0x10: 'BC2_SRGB', 0x11: 'BC3_SRGB',
}

class extractBFFNT (rawutil.TypeReader, ClsFunc):
	def main(self, filename, data, verbose, opts={}):
		self.verbose = verbose
		self.filebase = os.path.splitext(filename)[0]
		self.read_header(data)
		self.readFINF(data)
		self.glyphmap = {}
		self.readCMAP(data, self.cmapoffset - 8)
		self.glyphwidths = {}
		self.readCWDH(data, self.cwdhoffset - 8)
		self.readTGLP(data, self.tglpoffset - 8)
		meta = {'glyphmap': self.glyphmap, 'glyphwidths': self.glyphwidths}
		write(json.dumps(meta, indent=4), self.filebase + '_meta.json')
		
	
	def read_header(self, data):
		magic, bom = rawutil.unpack_from('>4sH', data, 0)
		if magic != b'FFNT':
			error.InvalidMagicError('Invalid magic %s, expected %s' % (byterepr(magic), 'FFNT'))
		self.byteorder = ENDIANS[bom]
		headerlen, version, filesize, blockcount = self.unpack_from('H3I', data)
		if version != 0x04000000:
			error.UnsupportedVersionError('Only version 4 is supported')
	
	def readFINF(self, data):
		magic, size = self.unpack_from('4sI', data)
		if magic != b'FINF':
			error.InvalidMagicError('Invalid FINF magic (got %s)' % byterepr(magic))
		self.fonttype, self.height, self.width, self.ascent, self.linefeed, self.alterindex = self.unpack_from('4B2H', data)
		self.leftwidth, self.glyphwidth, self.charwidth, self.encoding = self.unpack_from('4B', data)
		self.tglpoffset, self.cwdhoffset, self.cmapoffset = self.unpack_from('3I', data)
	
	def readTGLP(self, data, offset):
		magic, size = self.unpack_from('4sI', data, offset)
		if magic != b'TGLP':
			error.InvalidMagicError('Invalid TGLP magic (got %s)' % byterepr(magic))
		self.cellwidth, self.cellheight, self.sheetcount, self.maxwidth = self.unpack_from('4B', data)
		self.sheetsize, self.baselinepos, format = self.unpack_from('I2H', data)
		self.colcount, self.rowcount, self.sheetwidth, self.sheetheight, dataoffset = self.unpack_from('4HI', data)
		if self.byteorder == '<':
			self.format = FORMAT_NAMES_CTR[format]
		else:
			self.format = FORMAT_NAMES_CAFE[format]
		data.seek(dataoffset)
		for i in range(self.sheetcount):
			if self.sheetcount > 1:
				outname = self.filebase + '_sheet%d.png' % i
			else:
				outname = self.filebase + '.png'
			if c3DSkit is not None:
				self.extract_sheet_c3DSkit(data, outname)
			else:
				self.extract_sheet_py3DSkit(data, outname)
	
	def extract_sheet_c3DSkit(self, data, outname):
		out = np.ascontiguousarray(np.zeros(self.sheetwidth * self.sheetheight * 4, dtype=np.uint8))
		indata = np.ascontiguousarray(np.fromstring(data.read(self.sheetsize), dtype=np.uint8))
		format = c3DSkit.getTextureFormatId(self.format)
		c3DSkit.extractTiledTexture(indata, out, self.sheetwidth, self.sheetheight, format)
		Image.frombytes('RGBA', (self.sheetwidth, self.sheetheight), out.tostring()).save(outname, 'PNG')
	
	def extract_sheet_py3DSkit(self, data, outname):
		img = Image.new('RGBA', (self.sheetwidth, self.sheetheight))
		return img
	
	def readCWDH(self, data, secoffset):
		magic, size = self.unpack_from('4sI', data, secoffset)
		if magic != b'CWDH':
			error.InvalidMagicError('Invalid CWDH magic (got %s)' % byterepr(magic))
		startcode, endcode, nextoffset = self.unpack_from('2HI', data)
		for i, width in enumerate(self.unpack_from('%d[3b]' % (endcode - startcode), data)[0]):
			self.glyphwidths[chr(startcode + i)] = {'left': width[0], 'glyph': width[1], 'char': width[2]}
		if nextoffset > 0:
			self.readCWDH(data, nextoffset - 8)
	
	def readCMAP(self, data, secoffset):
		magic, size = self.unpack_from('4sI', data, secoffset)
		if magic != b'CMAP':
			error.InvalidMagicError('Invalid CMAP magic (got %s)' % byterepr(magic))
		startcode, endcode, method, reserved, nextoffset = self.unpack_from('4HI', data)
		if method == CMAP_DIRECT:
			indexoffset = self.unpack_from('H', data)[0]
			for code in range(startcode, endcode):
				self.glyphmap[chr(code)] = code - startcode + indexoffset
		elif method == CMAP_TABLE:
			for i, item in enumerate(self.unpack_from('%dH' % (endcode - startcode), data)):
				self.glyphmap[chr(i + startcode)] = item
		elif method == CMAP_SCAN:
			count, items = self.unpack_from('H/p1[2H]', data)
			for code, offset in items:
				self.glyphmap[chr(code)] = offset
		if nextoffset > 0:
			self.readCMAP(data, nextoffset - 8)