# -*- coding:utf-8 -*-
import math
from util import error, ENDIANS
import util.rawutil as rawutil
from util.funcops import ClsFunc
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
	def main(self, filename, data):
		self.outfile = make_outfile(filename, 'png')
		data = self.readheader(data)
		if self.format in (ETC1, ETC1A4, ETC1_2):
			self.decompress_ETC1(data)
		else:
			self.extract(data)
			
	def readheader(self, data):
		flim = data[-0x28:-0x14]
		imag = data[-0x14:]
		self.readFLIMheader(flim)
		self.readimagheader(imag)
		return data[:-0x28]
		
	def readFLIMheader(self, hdr):
		self.byteorder = ENDIANS[rawutil.unpack_from('>H', hdr, 4)[0]]
		hdata = self.unpack(BFLIM_FLIM_HDR_STRUCT, hdr)
		if hdata[0] != b'FLIM':
			error('Invalid magic: %s' % hdata[0])
		#bom = hdata[1]
		#headerlen = hdata[2]
		self.version = hdata[3]
		#filelen = hdata[4]
		#datablocksnum = hdata[5] #always 0x01
		#padding = hdata[6]
		
	def readimagheader(self, hdr):
		hdata = self.unpack(BFLIM_IMAG_HDR_STRUCT, hdr)
		if hdata[0] != b'imag':
			error('Invalid magic for imag header: %s' % hdata[0])
		#parsesize = hdata[1] #0x10
		self.width = hdata[2]
		self.height = hdata[3]
		self.align = hdata[4]
		self.format = hdata[5]
		self.swizzle = hdata[6]
		#datalen = hdata[7]
		
	def extract(self, data):
		img = Image.new('RGBA', (self.width, self.height))
		pixels = img.load()
		dataw = 1 << int(math.ceil(math.log(self.width, 2)))
		datah = 1 << int(math.ceil(math.log(self.height, 2)))
		tiles_width = math.ceil(self.width / 8)
		tiles_height = math.ceil(self.height / 8)
		for tiley in range(0, tiles_height):
			for tilex in range(0, tiles_width):
				for y1 in range(0, 2):
					for x1 in range(0, 2):
						for y2 in range(0, 2):
							for x2 in range(0, 2):
								for y3 in range(0, 2):
									for x3 in range(0, 2):
										pixel_x = (x3 + (x2 * 2) + (x1 * 4) + (tilex * 8))
										pixel_y = (y3 + (y2 * 2) + (y1 * 4) + (tiley * 8))
										if pixel_x >= self.width:
											continue
										if pixel_y >= self.height:
											continue
										data_x = (x3 + (x2 * 4) + (x1 * 16) + (tilex * 64))
										data_y = ((y3 * 2) + (y2 * 8) + (y1 * 32) + (tiley * dataw * 8))
										pos = data_x + data_y
										px = self.getpixel(data, pos)
										pixels[pixel_x, pixel_y] = px
		img = self.deswizzle(img)
		img.save(self.outfile, 'PNG')
		
	def getpixel(self, data, ptr):
		if self.format == L8:
			l = self.unpack_from('B', data, ptr)[0]
			px = (l, l, l, 255)
		elif self.format == A8:
			px = (0, 0, 0) + self.unpack_from('B', data, ptr)
		elif self.format == LA4:
			la = self.unpack_from('B', data, ptr)[0]
			l = (la >> 4) *0x11
			a = (la & 0x0f) * 0x11
			px = (l, l, l, a)
		elif self.format == LA8:
			l, a = self.unpack_from('2B', data, ptr)
			px = (l, l, l, a)
		elif self.format == RGB565:
			val = self.unpack_from('H', data, ptr)[0]
			r = (val >> 11)
			g = ((val & 0b0000011111100000) >> 5)
			b = (val & 0b0000000000011111)
			px = (r, g, b, 255)
		elif self.format == RGB8:
			px = self.unpack_from('3B', data, ptr) + (255,)
		elif self.format == RGBA5551:
			val = self.unpack_from('H', data, ptr)[0]
			r = (val >> 11) * 8
			g = ((val & 0b0000011111000000) >> 6) * 8
			b = ((val & 0b0000000000111110) >> 1) * 8
			a = (val & 1) * 255
			px = (r, g, b, a)
		elif self.format == RGBA8:
			px = self.unpack_from('4B', data, ptr)
		elif self.format == RGBA4:
			rg, ba = self.unpack_from('2B', data, ptr)
			r = (rg >> 4) * 0x11
			g = (rg & 0x0F) * 0x11
			b = (ba >> 4) * 0x11
			a = (ba & 0x01) * 0x11
			px = (r, g, b, a)
		elif self.format == L4:
			val = self.unpack_from('B', data, ptr // 2)[0]
			shift = (ptr & 1) * 4
			l = ((val >> shift) & 0x0f) * 0x11
			px = (l, l, l, 255)
		elif self.format == A4:
			val = self.unpack_from('B', data, ptr // 2)[0]
			shift = (ptr & 1) * 4
			a = ((val >> shift) & 0x0f) * 0x11
			px = (0, 0, 0, a)
		else:
			error('Unsupported texture format')
		return px
		
	def deswizzle(self, img):
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
		has_alpha = (self.format == ETC1A4)
		blklen = (16 if has_alpha else 8)
		img = Image.new('RGBA', (self.width, self.height))
		pixels = img.load()
		tile_w = int(math.ceil(self.width / 8))
		tile_h = int(math.ceil(self.height / 8))
		tile_w = 1 << int(math.ceil(math.log(tile_w, 2)))
		tile_h = 1 << int(math.ceil(math.log(tile_h, 2)))
		ptr = 0
		for tiley in range(0, tile_h):
			for tilex in range(0, tile_w):
				for blocky in range(0, 2):
					for blockx in range(0, 2):
						block = data[ptr:ptr + blklen]
						ptr += blklen
						if has_alpha:
							alphas = self.unpack_from('Q', block, 0)[0]
							block = block[8:]
						else:
							alphas = (2 ** 64) - 1  #0xffff... for 8B
						pxs = self.unpack('Q', block, 0)[0]
						differential = (pxs >> 33) & 1
						horizontal = (pxs >> 32) & 1
						table1 = ETC1_MODIFIERS[(pxs >> 37) & 7]
						table2 = ETC1_MODIFIERS[(pxs >> 34) & 7]
						if differential:
							r = (pxs >> 59) & 0x1f
							g = (pxs >> 51) & 0x1f
							b = (pxs >> 43) & 0x1f
							r = (r << 3) | ((r >> 2) & 7)
							g = (g << 3) | ((g >> 2) & 7)
							b = (b << 3) | ((b >> 2) & 7)
							color1 = (r, g, b)
							
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
						signs = pxs & 0xffff
						for pixely in range(0, 4):
							for pixelx in range(0, 4):
								x = pixelx + (blockx * 4) + (tilex * 8)
								y = pixely + (blocky * 4) + (tiley * 8)
								if x >= self.width:
									continue
								if y >= self.height:
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
