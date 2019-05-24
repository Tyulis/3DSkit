# -*- coding:utf-8 -*-
import json
from util import error
from util.utils import ClsFunc
from util.rawutil import TypeWriter
from util.filesystem import *

VERSIONS = {
	0x03000000: 'CTR',
	0x04000000: 'CAFE',
	0x04010000: 'NX',
}

FINF_OFFSET = 0x14
TGLP_OFFSET = 0x34
TEXTURE_DATA_OFFSET = 0x1000

FORMATS_NX = {
	'RGBA8': 0x00, 'RGB8': 0x01,
	'RGBA5551': 0x02, 'RGB565': 0x03,
	'RGBA4': 0x04, 'LA8': 0x05,
	'LA4': 0x06, 'A4': 0x07,
	'A8': 0x08, 'BC1': 0x09,
	'BC2': 0x0A, 'BC3': 0x0B,
	'BC4': 0x0C, 'BC5': 0x0D,
	'RGBA8_SRGB': 0x0E, 'BC1_SRGB': 0x0F,
	'BC2_SRGB': 0x10, 'BC3_SRGB': 0x11,
	'BC7': 0x12, 'BC7_SRGB': 0x13,
}

class packBFFNT(ClsFunc, TypeWriter):
	def main(self, filenames, outname, endian, verbose, opts={}):
		self.byteorder = endian
		self.verbose = verbose
		self.inname = filenames[0]
		with open(self.inname, 'r') as f:
			self.meta = json.load(f)
		out = open(outname, 'wb+')
		out.seek(TGLP_OFFSET)
		self.packTGLP(out, self.meta['textures'])
		self.packCWDH(out, self.meta['glyphwidths'])
		self.packCMAP(out, self.meta['glyphmap'])
		out.seek(FINF_OFFSET)
		self.packFINF(out, self.meta['info'])
		out.seek(0, 2)
		self.filesize = out.tell()
		out.seek(0)
		self.pack_header(out, self.meta)

	def pack_header(self, out, meta):
		self.pack('4s2H3I', b'FFNT', 0xFEFF, 0x14, meta['version_number'], self.filesize, 4, out)

	def packFINF(self, out, meta):
		self.pack('4sI 4B2H 4B3I', b'FINF', 32, meta['type'], meta['height'], meta['width'], meta['ascent'], meta['line_feed'], meta['alter_index'],
					meta['default_left_width'], meta['default_glyph_width'], meta['default_char_width'], meta['encoding'], self.tglpoffset + 8, self.cwdhoffset + 8, self.cmapoffset + 8, out)

	def packTGLP(self, out, meta):
		if self.meta['version'] != 'NX':
			error.UnsupportedVersionError('Only Switch BFFNTs can be repacked yet')
		if self.meta['version'] == 'NX':
			self.packTGLP_NX(out, meta)

	def packTGLP_NX(self, out, meta):
		self.tglpoffset = out.tell()
		try:
			bntx = open(meta['output'][0], 'rb')
		except FileNotFoundError:
			error.FileNotFoundError('The file %s could not be found. Check if it has the right name in the "textures > output" property in %s' % (meta['output'][0], self.inname))
		if bntx.read(4) != b'BNTX':
			error.InvalidInputError('%s is not a valid BNTX file' % meta['output'][0])
		bntx.seek(0, 2)
		bntxsize = bntx.tell()
		bntx.seek(0)
		format = FORMATS_NX[meta['format']]
		sheetsize = bntxsize // meta['num_sheets']
		self.pack('4s4x 4BI6H I', b'TGLP', meta['cell_width'], meta['cell_height'], meta['num_sheets'], meta['max_width'], sheetsize, meta['base_line_position'], format, meta['num_columns'], meta['num_lines'], meta['sheet_width'], meta['sheet_height'], TEXTURE_DATA_OFFSET, out)
		out.seek(TEXTURE_DATA_OFFSET)
		out.write(bntx.read())
		bntx.close()
		endpos = out.tell()
		out.seek(self.tglpoffset + 4)
		self.pack('I', endpos - self.tglpoffset, out)
		out.seek(endpos)

	def packCWDH(self, out, meta):
		# Naive implementation, to improve ?
		self.cwdhoffset = out.tell()
		startcode = ord(min(meta.keys()))
		endcode = ord(max(meta.keys()))
		secsize = 16 + 3 * (endcode - startcode)
		self.pack('4sI 2HI', b'CWDH', secsize, startcode, endcode, 0, out)
		for char in sorted(meta.keys()):
			self.pack('3b', meta[char]['left'], meta[char]['glyph'], meta[char]['char'], out)

	def packCMAP(self, out, meta):
		# Naive implementation, to improve ?
		self.cmapoffset = out.tell()
		startcode = ord(min(meta.keys()))
		endcode = ord(max(meta.keys()))
		secsize = 16 + (8 if self.meta['version'] == 'NX' else 4) + 2
		self.pack('4sI', b'CMAP', secsize, out)
		if self.meta['version'] == 'NX':
			self.pack('2I', startcode, endcode, out)
		else:
			self.pack('2H', startcode, endcode, out)
		self.pack('2H2I', 0, 0, 0, 0, out)
