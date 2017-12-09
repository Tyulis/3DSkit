# -*- coding:utf-8 -*-
import math
from util import error, ENDIANS
import util.rawutil as rawutil
from util.funcops import ClsFunc, byterepr
from util.fileops import *
from libkit.texinfo import *
from libkit.texextract import *
from PIL import Image

BCLIM_CLIM_HDR_STRUCT = '4s2H2I2H'
BCLIM_IMAG_HDR_STRUCT = '4sI2HI2H'

class extractBCLIM(ClsFunc, rawutil.TypeReader):
	def main(self, filename, data, verbose, opts={}):
		self.outfile = make_outfile(filename, 'png')
		self.verbose = verbose
		self.readheader(data)
		if self.format in (ETC1, ETC1A4, ETC1_2):
			if self.verbose:
				print('Decompressing pixel data')
			img = decompress_ETC1(data, self.width, self.height, self.format, self.byteorder)
		else:
			if self.verbose:
				print('Extracting pixel data')
			img = extract_texture(data, self.width, self.height, self.format, self.byteorder)
		if self.verbose and self.swizzle != 0:
			print('Deswizzling...')
		img = deswizzle(img, self.swizzle)
		img.save(self.outfile, 'PNG')
			
	def readheader(self, data):
		self.readCLIMheader(data)
		self.readimagheader(data)
		data.seek(0)
		
	def readCLIMheader(self, data):
		data.seek(-0x24, 2)
		self.byteorder = ENDIANS[rawutil.unpack_from('>H', data)[0]]
		data.seek(-0x28, 2)
		hdata = self.unpack_from(BCLIM_CLIM_HDR_STRUCT, data)
		if hdata[0] != b'CLIM':
			error.InvalidMagicError('Invalid magic %s, expected CLIM' % byterepr(hdata[0]))
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
		hdata = self.unpack_from(BCLIM_IMAG_HDR_STRUCT, data)
		if hdata[0] != b'imag':
			error.InvalidMagicError('Invalid magic for imag header: %s' % byterepr(hdata[0]))
		#headerlen = hdata[1] #0x10
		self.width = hdata[2]
		self.height = hdata[3]
		self.format = hdata[4]
		print('Color format: %s' % FORMAT_NAMES[self.format])
		self.pxsize = PIXEL_SIZES[self.format]
		self.swizzle = hdata[5]
		if self.verbose:
			print('Width: %d' % self.width)
			print('Height: %d' % self.height)
		#if self.swizzle in (4, 8) and self.version == 0x07020100:
		#	self.width, self.height = self.height, self.width
		print('Texture swizzling: %d' % self.swizzle)
		#datalen = hdata[7]
