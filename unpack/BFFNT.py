# -*- coding:utf-8 -*-
import os
import json
import numpy as np
from PIL import Image
from util import error, ENDIANS, libkit
from util.utils import byterepr, ClsFunc
from util.filesystem import *
import util.rawutil as rawutil

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
		if 'origin' in opts:
			self.version = opts['origin'].upper().strip()
		else:
			self.version = None
		if 'reverse' in opts:
			self.reverse = opts['reverse'].lower().strip() == 'true'
		else:
			self.reverse = False
		self.filebase = os.path.splitext(filename)[0]
		self.read_header(data)
		finf = self.readFINF(data)
		self.glyphmap = {}
		self.readCMAP(data, self.cmapoffset - 8)
		self.glyphwidths = {}
		self.readCWDH(data, self.cwdhoffset - 8)
		tglp = self.readTGLP(data, self.tglpoffset - 8)
		meta = {'version': self.version, 'version_number': self.vernum, 'info': finf, 'glyphmap': self.glyphmap, 'glyphwidths': self.glyphwidths, 'textures': tglp}
		write(json.dumps(meta, indent=4), self.filebase + '_meta.json')

	def read_header(self, data):
		magic, bom = rawutil.unpack_from('>4sH', data, 0)
		if magic != b'FFNT':
			error.InvalidMagicError('Invalid magic %s, expected %s' % (byterepr(magic), 'FFNT'))
		self.byteorder = ENDIANS[bom]
		headerlen, version, filesize, blockcount = self.unpack_from('H3I', data)
		if version not in (0x04010000, 0x04000000, 0x03000000):
			error.UnsupportedVersionError('Only versions 4.0.0, 4.1.0 and 3.0.0 are supported, found %d.%d.%d' % (version >> 24, (version >> 16) & 0xFF, version & 0xFFFF))
		if self.version is None:
			self.version = VERSIONS[version]
		self.vernum = version
		if self.verbose:
			print('Version %d.%d.%d (%s)' % (version >> 24, (version >> 16) & 0xFF, version & 0xFFFF, self.version))

	def readFINF(self, data):
		magic, size = self.unpack_from('4sI', data)
		if magic != b'FINF':
			error.InvalidMagicError('Invalid FINF magic (got %s)' % byterepr(magic))
		self.fonttype, self.height, self.width, self.ascent, self.linefeed, self.alterindex = self.unpack_from('4B2H', data)
		self.defaultleftwidth, self.defaultglyphwidth, self.defaultcharwidth, self.encoding = self.unpack_from('4B', data)
		self.tglpoffset, self.cwdhoffset, self.cmapoffset = self.unpack_from('3I', data)
		node = {'type': self.fonttype, 'height': self.height, 'width': self.width, 'ascent': self.ascent, 'line_feed': self.linefeed, 'alter_index': self.alterindex,
				'default_left_width': self.defaultleftwidth, 'default_glyph_width': self.defaultglyphwidth, 'default_char_width': self.defaultcharwidth,
				'encoding': self.encoding}
		return node

	def readTGLP(self, data, offset):
		magic, size = self.unpack_from('4sI', data, offset)
		if magic != b'TGLP':
			error.InvalidMagicError('Invalid TGLP magic (got %s)' % byterepr(magic))
		self.cellwidth, self.cellheight, self.sheetcount, self.maxwidth = self.unpack_from('4B', data)
		self.sheetsize, self.baselinepos, format = self.unpack_from('I2H', data)
		self.colcount, self.rowcount, self.sheetwidth, self.sheetheight, dataoffset = self.unpack_from('4HI', data)
		node = {'cell_width': self.cellwidth, 'cell_height': self.cellheight, 'num_sheets': self.sheetcount, 'max_width': self.maxwidth,
				'base_line_position': self.baselinepos, 'num_columns': self.colcount, 'num_lines': self.rowcount, 'sheet_width': self.sheetwidth, 'sheet_height': self.sheetheight}
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
		node['format'] = self.format
		if self.verbose:
			print('Number of sheets: %d' % self.sheetcount)
			print('Texture format: %s' % self.format)
			print('Sheet width: %d' % self.sheetwidth)
			print('Sheet height: %d' % self.sheetheight)
		data.seek(dataoffset)
		magic = data.read(4)
		data.seek(-4, 1)
		if (magic == b'BNTX'):
			node['output'] = [self.extract_underlying_BNTX(data)]
			node['BNTX'] = True
			return node
		texfilenames = []
		for i in range(self.sheetcount):
			if self.sheetcount > 1:
				outname = self.filebase + '_sheet%d.png' % i
			else:
				outname = self.filebase + '.png'
			rgba = self.extract_sheet(data, self.sheetwidth, self.sheetheight, self.sheetsize, self.format, 16)
			sheet = Image.frombytes('RGBA', (self.sheetwidth, self.sheetheight), rgba.tostring())
			if self.reverse:
				sheet = sheet.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
			sheet.save(outname, 'PNG')
			texfilenames.append(outname)
		node['output'] = texfilenames
		node['BNTX'] = False
		return node

	def extract_sheet(self, data, width, height, size, format, swizzlesize):
		out = np.ascontiguousarray(np.zeros(width * height * 4, dtype=np.uint8))
		indata = np.ascontiguousarray(np.fromstring(data.read(size), dtype=np.uint8))
		formatid = libkit.getTextureFormatId(format)
		if formatid == -1:
			error.UnsupportedDataFormatError('%s texture format is not supported yet' % format)
		libkit.extractTiledTexture(indata, out, width, height, formatid, swizzlesize, self.byteorder == '<')
		return out

	"""def extract_underlying_BNTX(self, data):
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
		#print('Number of sheets: %d' % sheetcount)
		for texindex in range(sheetcount):
			if sheetcount > 1:
				outbase = '%s_tex%d' % (self.filebase, texindex)
			else:
				outbase = self.filebase
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
				print(' - Swizzle size: %d' % swizzlesize)
			dataoffset = self.unpack_from('Q', data, bntxpos + pointersoffset)[0]
			data.seek(bntxpos + dataoffset)
			rgba = self.extract_sheet(data, self.sheetwidth, self.sheetheight * self.sheetcount, sheetsize, BNTI_FORMAT_NAMES[format], swizzlesize)
			for i in range(self.sheetcount):
				if self.sheetcount > 1:
					outname = outbase + '_sheet%d.png' % i
				else:
					outname = outbase + '.png'
				sheet = rgba[4 * self.sheetwidth * i * self.sheetheight: 4 * self.sheetwidth * (i + 1) * self.sheetheight]
				img = Image.frombytes('RGBA', (self.sheetwidth, self.sheetheight), sheet.tostring())
				if self.reverse:
					img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
				img.save(outname, 'PNG')"""

	def extract_underlying_BNTX(self, data):
		bntxpos = data.tell()
		magic, version, bom, revision, nameoffset, stroffset, relocoffset, filesize = self.unpack_from('8sI2H4I', data)
		data.seek(bntxpos)
		bntxname = self.filebase + '_texture.bntx'
		with open(bntxname, 'wb') as f:
			f.write(data.read(filesize))
		return bntxname

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
