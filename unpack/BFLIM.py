# -*- coding:utf-8 -*-
import math
from util import error, ENDIANS, libkit
import util.rawutil as rawutil
from util.utils import ClsFunc, byterepr
from util.filesystem import *
from PIL import Image
import numpy as np

BFLIM_FLIM_HDR_STRUCT = '4s2H2I2H'
BFLIM_IMAG_HDR_STRUCT = '4sI3H2BI'

#BFLIM formats
L8 = 0x00
A8 = 0x01
LA4 = 0x02
LA8 = 0x03
HILO8 = 0x04
RGB565 = 0x05
RGB8 = 0x06
RGBA5551 = 0x07
RGBA4 = 0x08
RGBA8 = 0x09
ETC1 = 0x0a
ETC1A4 = 0x0b
L4 = 0x0c
A4 = 0x0d
ETC1_2 = 0x13

PIXEL_SIZES = {
	L8: 1,
	A8: 1,
	LA4: 1,
	LA8: 2,
	HILO8: 2,
	RGB565: 2,
	RGB8: 3,
	RGBA5551: 2,
	RGBA4: 2,
	RGBA8: 4,
	ETC1: None,
	ETC1A4: None,
	L4: 0.5,
	A4: 0.5,
	ETC1_2: None
}

FORMAT_NAMES = {
	L8: 'L8',
	A8: 'A8',
	LA4: 'LA4',
	LA8: 'LA8',
	HILO8: 'HILO8',
	RGB565: 'RGB565',
	RGB8: 'RGB8',
	RGBA5551: 'RGBA5551',
	RGBA4: 'RGBA4',
	RGBA8: 'RGBA8',
	ETC1: 'ETC1',
	ETC1A4: 'ETC1A4',
	L4: 'L4',
	A4: 'A4',
	ETC1_2: 'ETC1',
}

ETC1_MODIFIERS = [
	[2, 8],
	[5, 17],
	[9, 29],
	[13, 42],
	[18, 60],
	[24, 80],
	[33, 106],
	[47, 183]
]

class extractBFLIM(ClsFunc, rawutil.TypeReader):
	def main(self, filename, data, verbose, opts={}):
		self.outfile = make_outfile(filename, 'png')
		self.verbose = verbose
		self.readheader(data)
		self.extract(data)
			
	def readheader(self, data):
		self.readFLIMheader(data)
		self.readimagheader(data)
		data.seek(0)
		
	def readFLIMheader(self, data):
		data.seek(-0x24, 2)
		self.byteorder = ENDIANS[rawutil.unpack_from('>H', data)[0]]
		data.seek(-0x28, 2)
		hdata = self.unpack_from(BFLIM_FLIM_HDR_STRUCT, data)
		if hdata[0] != b'FLIM':
			error.InvalidMagicError('Invalid magic %s, expected FLIM' % byterepr(hdata[0]))
		#bom = hdata[1]
		#headerlen = hdata[2]
		self.version = hdata[3]
		if self.verbose:
			print('Version: %08x' % self.version)
		#filelen = hdata[4]
		#datablocksnum = hdata[5] #always 0x01
		#padding = hdata[6]
		
	def readimagheader(self, data):
		data.seek(-0x14, 2)
		hdata = self.unpack_from(BFLIM_IMAG_HDR_STRUCT, data)
		if hdata[0] != b'imag':
			error.InvalidMagicError('Invalid magic for imag header: %s' % byterepr(hdata[0]))
		#headerlen = hdata[1] #0x10
		self.width = hdata[2]
		self.height = hdata[3]
		self.align = hdata[4]
		self.format = hdata[5]
		print('Color format: %s' % FORMAT_NAMES[self.format])
		self.pxsize = PIXEL_SIZES[self.format]
		self.swizzle = hdata[6]
		if self.verbose:
			print('Width: %d' % self.width)
			print('Height: %d' % self.height)
		if self.swizzle in (4, 8) and self.version == 0x07020100:
			self.width, self.height = self.height, self.width
		print('Texture swizzling: %d' % self.swizzle)
		#datalen = hdata[7]
	
	def extract(self, data):
		format = libkit.getTextureFormatId(FORMAT_NAMES[self.format])
		if format == -1:
			error.UnsupportedDataFormatError('Unsupported texture format %s' % FORMAT_NAMES[self.format])
		indata = np.ascontiguousarray(np.fromstring(data.read(), dtype=np.uint8))
		out = np.ascontiguousarray(np.zeros(self.width * self.height * 4, dtype=np.uint8))
		libkit.extractTiledTexture(indata, out, self.width, self.height, format, -1, self.byteorder == '<')
		img = Image.frombytes('RGBA', (self.width, self.height), out.tostring())
		img = self.deswizzle(img)
		img.save(self.outfile, 'PNG')
		
	def deswizzle(self, img):
		if self.verbose and self.swizzle != 0:
			print('Deswizzling')
		if self.swizzle == 4:
			img = img.transpose(Image.ROTATE_90)
		elif self.swizzle == 8:
			img = img.transpose(Image.ROTATE_90)
			img = img.transpose(Image.FLIP_TOP_BOTTOM)
		return img