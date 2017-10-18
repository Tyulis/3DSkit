# -*- coding:utf-8 -*-
import math
from util.funcops import ClsFunc
from util.rawutil import TypeWriter
from util.fileops import *
from PIL import Image

BFLIM_FLIM_HDR_STRUCT = '4s2H2I2H'
BFLIM_IMAG_HDR_STRUCT = '4sI3H2BI'

ALIGNMENT = 0x80
VERSION = 0x07020100

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

FORMATS = {
	'L8': L8,
	'A8': A8,
	'LA4': LA4,
	'LA8': LA8,
	'HILO8': HILO8,
	'RGB565': RGB565,
	'RGB8': RGB8,
	'RGBA5551': RGBA5551,
	'RGBA4': RGBA4,
	'RGBA8': RGBA8,
	'ETC1': ETC1,
	'ETC1A4': ETC1A4,
	'L4': L4,
	'A4': A4,
	'ETC1_2': ETC1_2
}


class packBFLIM(ClsFunc, TypeWriter):
	def main(self, filenames, outname, endian, opts={}):
		self.byteorder = endian
		filename = filenames[0]
		img = Image.open(filename)
		if 'format' in opts.keys():
			self.format = FORMATS[opts['format'].upper()]
		else:
			self.format = RGBA8
		img = self.swizzle(img, opts['swizzle'])
		data = self.repack_data(img)
		final = data
		final += self.align(final, ALIGNMENT)
		filelen = len(final) + 0x28
		flimhdr = self.repackFLIMheader(filelen)
		imaghdr = self.repackIMAGheader(img, int(opts['swizzle']), len(data))
		final += flimhdr + imaghdr
		bwrite(final, outname)
	
	def repackFLIMheader(self, filelen):
		#bom always 0xfeff, because 0xfeff packed in little endian gives 0xfffe, avoid some ugly lines
		return self.pack(BFLIM_FLIM_HDR_STRUCT, b'FLIM', 0xfeff, 0x14, VERSION, filelen, 1, 0)
	
	def repackIMAGheader(self, img, swizzle, datalen):
		width, height = img.size
		return self.pack(BFLIM_IMAG_HDR_STRUCT, b'imag', 0x10, width, height, ALIGNMENT, self.format, swizzle, datalen)
	
	def swizzle(self, img, swizzle):
		swizzle = int(swizzle)
		if swizzle == 4:
			img = img.rotate(-90)
		elif swizzle == 8:
			img = img.transpose(Image.FLIP_LEFT_RIGHT)
			img = img.rotate(-90)
		return img
	
	def repack_data(self, img):
		self.pxsize = PIXEL_SIZES[self.format]
		pixels = img.load()
		width, height = img.size
		datawidth = 1 << int(math.ceil(math.log(width, 2)))
		dataheight = 1 << int(math.ceil(math.log(height, 2)))
		tiles_x = math.ceil(datawidth / 8)
		tiles_y = math.ceil(dataheight / 8)
		final = bytearray(datawidth * dataheight)
		tilelen = 64 * self.pxsize
		#pos = 0
		for ytile in range(tiles_y):
			for xtile in range(tiles_x):
				tile = self.pack_tile(pixels, xtile, ytile, width, height)
				pos = ((ytile * tiles_x) + xtile) * tilelen
				final[pos:pos + tilelen] = tile
				#pos += tilelen
		return final
	
	def pack_tile(self, pixels, xtile, ytile, width, height):
		sublen = 16 * self.pxsize
		grouplen = 4 * self.pxsize
		tile = bytearray(64 * self.pxsize)
		for ysub in range(2):
			for xsub in range(2):
				sub = bytearray(16 * self.pxsize)
				for ygroup in range(2):
					for xgroup in range(2):
						group = bytearray(4 * self.pxsize)
						for ypix in range(2):
							for xpix in range(2):
								posy = (ytile * 8) + (ysub * 4) + (ygroup * 2) + ypix
								posx = (xtile * 8) + (xsub * 4) + (xgroup * 2) + xpix
								if posx >= width:
									rgba = (0, 0, 0, 0)
								elif posy >= height:
									rgba = (0, 0, 0, 0)
								else:
									rgba = pixels[posx, posy]
								packed = self.pack_pixel(rgba)
								pos = ((ypix * 2) + xpix) * self.pxsize
								group[pos:pos + self.pxsize] = packed
						pos = ((ygroup * 8) + (xgroup * 4)) * self.pxsize
						sub[pos:pos + grouplen] = group
				pos = ((ysub * 32) + (xsub * 16)) * self.pxsize
				tile[pos:pos + sublen] = sub
		return tile

	def pack_pixel(self, rgba):
		r, g, b, a = rgba
		if self.format == L8:
			l = math.ceil((r * 0.2126) + (g * 0.7152) + (b * 0.0722))
			return self.pack('B', l)
		elif self.format == A8:
			return self.pack('B', a)
		elif self.format == LA4:
			l = math.ceil((r * 0.2126) + (g * 0.7152) + (b * 0.0722)) // 0x11
			la = (l << 4) + (a // 0x11)
			return self.pack('B', la)
		elif self.format == LA8:
			l = math.ceil((r * 0.2126) + (g * 0.7152) + (b * 0.0722))
			return self.pack('2B', l, a)
		elif self.format == HILO8:
			pass
		elif self.format == RGB8:
			return self.pack('3B', r, g, b)
		elif self.format == RGB565:
			r = (r // 8) << 11
			g = (g // 4) << 5
			b = (b // 8)
			return self.pack('H', r + g + b)
		elif self.format == RGBA5551:
			a = int(a != 0)
			r = (r // 8) << 11
			g = (g // 8) << 6
			b = (b // 8) << 1
			return self.pack('H', r + g + b + a)
		elif self.format == RGBA4:
			r = (r // 0x11) << 12
			g = (g // 0x11) << 8
			b = (b // 0x11) << 4
			a = (a // 0x11)
			return self.pack('H', r + g + b + a)
		elif self.format == RGBA8:
			return self.pack('4B', r, g, b, a)
		
