# -*- coding:utf-8 -*-
import math
from util import error, ENDIANS
import util.rawutil as rawutil
from util.funcops import ClsFunc, byterepr
from util.fileops import *
from PIL import Image

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
		if self.format in (ETC1, ETC1A4, ETC1_2):
			self.decompress_ETC1(data)
		else:
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
			error('Invalid magic %s, expected FLIM' % byterepr(hdata[0]), 301)
		#bom = hdata[1]
		#headerlen = hdata[2]
		self.version = hdata[3]
		#filelen = hdata[4]
		#datablocksnum = hdata[5] #always 0x01
		#padding = hdata[6]
		
	def readimagheader(self, data):
		data.seek(-0x14, 2)
		hdata = self.unpack_from(BFLIM_IMAG_HDR_STRUCT, data)
		if hdata[0] != b'imag':
			error('Invalid magic for imag header: %s' % byterepr(hdata[0]), 301)
		#headerlen = hdata[1] #0x10
		self.width = hdata[2]
		self.height = hdata[3]
		self.align = hdata[4]
		self.format = hdata[5]
		print('Color format: %s' % FORMAT_NAMES[self.format])
		self.pxsize = PIXEL_SIZES[self.format]
		self.swizzle = hdata[6]
		print('Texture swizzling: %d' % self.swizzle)
		#datalen = hdata[7]
		if self.verbose:
			print('Width: %d' % self.width)
			print('Height: %d' % self.height)
		
	def extract(self, data):
		if self.verbose:
			print('Extracting pixel data')
		img = Image.new('RGBA', (self.width, self.height))
		self.pixels = img.load()
		
		tiles_x = math.ceil(self.width / 8)
		tiles_y = math.ceil(self.height / 8)
		tilelen = int(64 * self.pxsize)
		
		#8x8px tiles
		for ytile in range(tiles_y):
			for xtile in range(tiles_x):
				tilepos = int(((ytile * 64 * tiles_x) + (xtile * 64)) * self.pxsize)
				data.seek(tilepos)
				tile = data.read(tilelen)
				self.extract_tile(tile, xtile, ytile)
				
		img = self.deswizzle(img)
		img.save(self.outfile, 'PNG')
		
	def extract_tile(self, tile, xtile, ytile):
		for ysub in range(2):
			for xsub in range(2):
				subpos = int(((ysub * 32) + (xsub * 16)) * self.pxsize)
				sub = tile[subpos:int(subpos + 16 * self.pxsize)]
				for ygroup in range(2):
					for xgroup in range(2):
						grppos = int(((ygroup * 8) + (xgroup * 4)) * self.pxsize)
						grp = sub[grppos:int(grppos + 4 * self.pxsize)]
						if self.pxsize == 0.5:
							for ypix in range(2):
								xpix = grp[ypix:ypix + 1]
								px1 = self.getpixel(xpix, subpx=1)
								px2 = self.getpixel(xpix, subpx=0)
								outpos_y = (ytile * 8) + (ysub * 4) + (ygroup * 2) + ypix
								outpos_x = (xtile * 8) + (xsub * 4) + (xgroup * 2)
								if outpos_x >= self.width or outpos_y >= self.height:
									continue
								self.pixels[outpos_x, outpos_y] = px1
								self.pixels[outpos_x + 1, outpos_y] = px2
						else:
							for ypix in range(2):
								for xpix in range(2):
									pixpos = ((ypix * 2) + xpix) * self.pxsize
									pixel = grp[pixpos:pixpos + self.pxsize]
									rgba = self.getpixel(pixel)
									outpos_y = (ytile * 8) + (ysub * 4) + (ygroup * 2) + ypix
									outpos_x = (xtile * 8) + (xsub * 4) + (xgroup * 2) + xpix
									if outpos_y >= self.height or outpos_x >= self.width:
										continue
									self.pixels[outpos_x, outpos_y] = rgba
									#assert rgba[3] ==  0
									
	def getpixel(self, data, ptr=0, subpx=0):
		if self.format == L8:
			l = self.unpack_from('B', data, ptr)[0]
			px = (l, l, l, 255)
		elif self.format == A8:
			px = [0, 0, 0] + self.unpack_from('B', data, ptr)
		elif self.format == LA4:
			la = self.unpack_from('B', data, ptr)[0]
			l = (la >> 4) * 0x11
			a = (la & 0x0f) * 0x11
			px = (l, l, l, a)
		elif self.format == LA8:
			l, a = self.unpack_from('2B', data, ptr)
			px = (l, l, l, a)
		elif self.format == RGB565:
			val = self.unpack_from('H', data, ptr)[0]
			r = int((val >> 11) * 8.225806451612904)
			g = int(((val & 0b0000011111100000) >> 5) * 4.0476190476190474)
			b = int((val & 0b0000000000011111) * 8.225806451612904)
			px = (r, g, b, 255)
		elif self.format == RGB8:
			px = self.unpack_from('3B', data, ptr) + [255]
		elif self.format == RGBA5551:
			val = self.unpack_from('H', data, ptr)[0]
			r = int((val >> 11) * 8.225806451612904)
			g = int(((val & 0b0000011111000000) >> 6) * 8.225806451612904)
			b = int(((val & 0b0000000000111110) >> 1) * 8.225806451612904)
			a = (val & 1) * 255
			px = (r, g, b, a)
		elif self.format == RGBA8:
			px = self.unpack_from('4B', data, ptr)
		elif self.format == RGBA4:
			rg, ba = self.unpack_from('2B', data, ptr)
			r = (rg >> 4) * 0x11
			g = (rg & 0x0F) * 0x11
			b = (ba >> 4) * 0x11
			a = (ba & 0x0F) * 0x11
			px = (r, g, b, a)
		elif self.format == L4:
			val = self.unpack_from('B', data, ptr)[0]
			shift = subpx * 4
			l = ((val >> shift) & 0x0f) * 0x11
			px = (l, l, l, 255)
		elif self.format == A4:
			val = self.unpack_from('B', data, ptr // 2)[0]
			shift = (ptr & 1) * 4
			a = ((val >> shift) & 0x0f) * 0x11
			px = (0, 0, 0, a)
		else:
			error('Unsupported texture format')
		return tuple(px)
		
	def deswizzle(self, img):
		if self.verbose and self.swizzle != 0:
			print('Deswizzling')
		if self.swizzle == 4:
			img = img.rotate(90)
		elif self.swizzle == 8:
			img = img.rotate(90)
			img = img.transpose(Image.FLIP_LEFT_RIGHT)
		return img
		
	def diff_complement(self, val, bits):
		if val >> (bits - 1) == 0:
			return val
		return val - (1 << bits)
		
	def decompress_ETC1(self, data):
		#Based on ObsidianX's BFLIM ETC1 decompression algorithm (https://github.com/ObsidianX/3dstools)
		if self.verbose:
			print('Decompressing pixel data')
		has_alpha = (self.format == ETC1A4)
		blklen = (16 if has_alpha else 8)
		img = Image.new('RGBA', (self.height, self.width))
		pixels = img.load()
		tile_h = math.ceil(self.width / 8)
		tile_w = math.ceil(self.height / 8)
		tile_h = 1 << math.ceil(math.log(tile_h, 2))
		tile_w = 1 << math.ceil(math.log(tile_w, 2))
		for tiley in range(0, tile_h):
			for tilex in range(0, tile_w):
				for blocky in range(0, 2):
					for blockx in range(0, 2):
						block = data.read(blklen)
						if has_alpha:
							alphas = self.unpack_from('Q', block, 0)[0]
							block = block[8:]
						else:
							alphas = (2 ** 64) - 1  #0xffff... on 8B
						pxs = self.unpack('Q', block)[0]
						differential = (pxs >> 33) & 1
						horizontal = (pxs >> 32) & 1
						table1 = ETC1_MODIFIERS[(pxs >> 37) & 7]
						table2 = ETC1_MODIFIERS[(pxs >> 34) & 7]
						if differential:
							r = (pxs >> 59) & 0x1f
							g = (pxs >> 51) & 0x1f
							b = (pxs >> 43) & 0x1f
							r1 = (r << 3) | ((r >> 2) & 7)
							g1 = (g << 3) | ((g >> 2) & 7)
							b1 = (b << 3) | ((b >> 2) & 7)
							color1 = (r1, g1, b1)
							
							r += self.diff_complement((pxs >> 56) & 7, 3)
							g += self.diff_complement((pxs >> 48) & 7, 3)
							b += self.diff_complement((pxs >> 40) & 7, 3)
							r = (r << 3) | ((r >> 2) & 7)
							g = (g << 3) | ((g >> 2) & 7)
							b = (b << 3) | ((b >> 2) & 7)
							color2 = (r, g, b)
						else:
							r = ((pxs >> 60) & 0x0f) * 0x11
							g = ((pxs >> 52) & 0x0f) * 0x11
							b = ((pxs >> 44) & 0x0f) * 0x11
							color1 = (r, g, b)
							
							r = ((pxs >> 56) & 0x0f) * 0x11
							g = ((pxs >> 48) & 0x0f) * 0x11
							b = ((pxs >> 40) & 0x0f) * 0x11
							color2 = (r, g, b)
						amounts = pxs & 0xffff
						signs = (pxs >> 16) & 0xffff
						for pixely in range(0, 4):
							for pixelx in range(0, 4):
								x = pixelx + (blockx * 4) + (tilex * 8)
								y = pixely + (blocky * 4) + (tiley * 8)
								if x >= self.height:
									continue
								if y >= self.width:
									continue
								offset = (pixelx * 4) + pixely
								if horizontal:
									table = table1 if pixely < 2 else table2
									color = color1 if pixely < 2 else color2
								else:
									table = table1 if pixelx < 2 else table2
									color = color1 if pixelx < 2 else color2
								amount = table[(amounts >> offset) & 1]
								sign = (signs >> offset) & 1
								if sign == 1:
									amount = -amount
								r = max(min(color[0] + amount, 0xFF), 0)
								g = max(min(color[1] + amount, 0xFF), 0)
								b = max(min(color[2] + amount, 0xFF), 0)
								a = ((alphas >> (offset * 4)) & 0x0F) * 0x11
								pixels[x, y] = (r, g, b, a)
		img = self.deswizzle(img)
		img.save(self.outfile, 'PNG')
