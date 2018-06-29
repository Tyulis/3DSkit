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

VERSIONS = {
	0x03000000: 'CTR',
	0x04000000: 'CAFE',
	0x04010000: 'NX',
}

FORMAT_NAMES_CTR = {
	0x00: 'RGBA8', 0x01: 'RGB8',
	0x02: 'RGBA5551', 0x03: 'RGB565',
	0x04: 'RGBA4', 0x05: 'LA8',
	0x06: 'RG8', 0x07: 'L8',
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

FORMAT_NAMES_NX = {
	0x00: 'RGBA8', 0x01: 'RGB8',
	0x02: 'RGBA5551', 0x03: 'RGB565',
	0x04: 'RGBA4', 0x05: 'LA8',
	0x06: 'LA4', 0x07: 'A4',
	0x08: 'A8', 0x09: 'BC1',
	0x0A: 'BC2', 0x0B: 'BC3',
	0x0C: 'BC4', 0x0D: 'BC5',
	0x0E: 'RGBA8_SRGB', 0x0F: 'BC1_SRGB',
	0x10: 'BC2_SRGB', 0x11: 'BC3_SRGB',
	0x12: 'BC7', 0x13: 'BC7_SRGB',
}

BNTI_FORMAT_NAMES = {
	0x0b01: 'RGBA8', 0x0b06: 'RGBA8_SRGB',
	0x0701: 'RGB565', 0x0201: 'A8', 0x0901: 'RG8',
	0x1a01: 'BC1', 0x1a06: 'BC1_SRGB',
	0x1b01: 'BC2', 0x1b06: 'BC2_SRGB',
	0x1c01: 'BC3', 0x1c06: 'BC3_SRGB',
	0x1d01: 'BC4', 0x1d02: 'BC4_SNORM',
	0x1e01: 'BC5', 0x1e02: 'BC5_SNORM',
	0x1f01: 'BC6H', 0x1f02: 'BC6H_SF16',
	0x2001: 'BC7', 0x2006: 'BC7_SRGB',
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
		if version not in (0x04010000, 0x04000000, 0x03000000):
			error.UnsupportedVersionError('Only versions 4.0.0, 4.1.0 and 3.0.0 are supported, found %d.%d.%d' % (version >> 24, (version >> 16) & 0xFF, version & 0xFFFF))
		self.version = VERSIONS[version]
		if self.verbose:
			print('Version %d.%d.%d (%s)' % (version >> 24, (version >> 16) & 0xFF, version & 0xFFFF, self.version))
	
	def readFINF(self, data):
		magic, size = self.unpack_from('4sI', data)
		if magic != b'FINF':
			error.InvalidMagicError('Invalid FINF magic (got %s)' % byterepr(magic))
		self.fonttype, self.height, self.width, self.ascent, self.linefeed, self.alterindex = self.unpack_from('4B2H', data)
		self.defaultleftwidth, self.defaultglyphwidth, self.defaultcharwidth, self.encoding = self.unpack_from('4B', data)
		self.tglpoffset, self.cwdhoffset, self.cmapoffset = self.unpack_from('3I', data)
	
	def readTGLP(self, data, offset):
		magic, size = self.unpack_from('4sI', data, offset)
		if magic != b'TGLP':
			error.InvalidMagicError('Invalid TGLP magic (got %s)' % byterepr(magic))
		self.cellwidth, self.cellheight, self.sheetcount, self.maxwidth = self.unpack_from('4B', data)
		self.sheetsize, self.baselinepos, format = self.unpack_from('I2H', data)
		self.colcount, self.rowcount, self.sheetwidth, self.sheetheight, dataoffset = self.unpack_from('4HI', data)
		'''elif self.version == 'CTR':
			self.cellwidth, self.cellheight, self.baselinepos, self.maxwidth = self.unpack_from('4B', data)
			self.sheetsize, self.sheetcount, format, self.colcount, self.rowcount = self.unpack_from('I4H', data)
			self.sheetwidth, self.sheetheight, dataoffset = self.unpack_from('2HI', data)'''
		if self.version == 'CTR':
			self.format = FORMAT_NAMES_CTR[format]
		elif self.version == 'CAFE':
			self.format = FORMAT_NAMES_CAFE[format]
		elif self.version == 'NX':
			self.format = FORMAT_NAMES_NX[format]
		if self.verbose:
			print('Texture format: %s' % self.format)
			print('Sheet width: %d' % self.sheetwidth)
			print('Sheet height: %d' % self.sheetheight)
		data.seek(dataoffset)
		magic = data.read(4)
		data.seek(-4, 1)
		if (magic == b'BNTX'):
			self.extract_underlying_BNTX(data)
			return
		for i in range(self.sheetcount):
			if self.sheetcount > 1:
				outname = self.filebase + '_sheet%d.png' % i
			else:
				outname = self.filebase + '.png'
			if c3DSkit is not None:
				self.extract_sheet_c3DSkit(data, outname, self.sheetwidth, self.sheetheight, self.sheetsize, self.format, 16)
			else:
				self.extract_sheet_py3DSkit(data, outname, self.sheetwidth, self.sheetheight, self.sheetsize, self.format, 16)
	
	def extract_sheet_c3DSkit(self, data, outname, width, height, size, format, swizzlesize):
		out = np.ascontiguousarray(np.zeros(width * height * 4, dtype=np.uint8))
		indata = np.ascontiguousarray(np.fromstring(data.read(size), dtype=np.uint8))
		format = c3DSkit.getTextureFormatId(format)
		if format == 0xFF:
			error.UnsupportedDataFormatError('%s texture format is not supported yet' % format)
		c3DSkit.extractTiledTexture(indata, out, width, height, format, swizzlesize, self.byteorder == '<')
		Image.frombytes('RGBA', (width, height), out.tostring()).save(outname, 'PNG')
	
	def extract_sheet_py3DSkit(self, data, outname, width, height, size, format, swizzlesize):
		error.NotImplementedError('You need c3DSkit to extract BFFNT sheets')
	
	def extract_underlying_BNTX(self, data):
		if self.verbose:
			print('\nExtracting wrapped BNTX')
		bntxpos = data.tell()
		# BNTX Header
		magic, unk4, version, bom, revision = self.unpack_from('4s2I2H', data)
		nameoffset, stroffset, relocoffset, filesize = self.unpack_from('4I', data)
		# NX   Header
		magic = data.read(4)
		if magic != b'NX  ':
			error.InvalidMagicError('Invalid NX   magic, got %s' % byterepr(magic))
		sheetcount, infoptroffset, dataoffset, dictoffset, strdictsize = self.unpack_from('I3QI', data)
		for texindex in range(sheetcount):
			if sheetcount > 1:
				outname = self.filebase + '_sheet%d.png' % texindex
			else:
				outname = self.filebase + '.png'
			brtioffset = self.unpack_from('Q', data, bntxpos + infoptroffset + texindex * 8)[0]
			# BRTI Header
			magic, size, longsize = self.unpack_from('4sIQ', data, bntxpos + brtioffset)
			if magic != b'BRTI':
				error.InvalidMagicError('Invalid BRTI magic, got %s' % byterepr(magic))
			tilemode, dim, flags, swizzlesize, mipmapnum, unk0x18, format, unk0x20 = self.unpack_from('2B3H3I', data)
			width, height, unk0x2C, facesnum, sizerange = self.unpack_from('5I', data)
			sheetsize, alignment, compsel, type, nameoffset, parentoffset, pointersoffset = self.unpack_from('24x4I3Q', data)
			if self.verbose:
				print('BNTI info for sheet %d:' % texindex)
				print(' - Format: %s' % BNTI_FORMAT_NAMES[format])
				print(' - Width: %d\n - Height: %d' % (width, height))
			dataoffset = self.unpack_from('Q', data, bntxpos + pointersoffset)[0]
			data.seek(bntxpos + dataoffset)
			if c3DSkit is not None:
				self.extract_sheet_c3DSkit(data, outname, width, height, sheetsize, BNTI_FORMAT_NAMES[format], swizzlesize)
			else:
				self.extract_sheet_py3DSkit(data, outname, width, height, sheetsize, BNTI_FORMAT_NAMES[format], swizzlesize)
	
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
		if self.version == 'NX':
			startcode, endcode, method, reserved, nextoffset = self.unpack_from('2I2HI', data)
		else:
			startcode, endcode, method, reserved, nextoffset = self.unpack_from('4HI', data)
		if method == CMAP_DIRECT:
			indexoffset = self.unpack_from('H', data)[0]
			for code in range(startcode, endcode):
				self.glyphmap[chr(code)] = code - startcode + indexoffset
		elif method == CMAP_TABLE:
			for i, item in enumerate(self.unpack_from('%dH' % (endcode - startcode), data)):
				self.glyphmap[chr(i + startcode)] = item
		elif method == CMAP_SCAN:
			if self.version == 'NX':
				count, items = self.unpack_from('I/p1[IH2x]', data)
			else:
				count, items = self.unpack_from('H/p1[2H]', data)
			for code, offset in items:
				self.glyphmap[chr(code)] = offset
		if nextoffset > 0:
			self.readCMAP(data, nextoffset - 8)
