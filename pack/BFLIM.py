# -*- coding:utf-8 -*-
import math
from util.funcops import ClsFunc
from util.rawutil import TypeWriter
from util.fileops import *
from PIL import Image

BFLIM_FLIM_HDR_STRUCT = '4s2H2I2H'
BFLIM_IMAG_HDR_STRUCT = '4sI3H2BI'

ALIGNMENT = 0x80
VERSION = 0x07020000

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
	def main(self, filenames, outname, endian, verbose, opts={}):
		self.byteorder = endian
		self.verbose = verbose
		filename = filenames[0]
		img = Image.open(filename)
		self.width, self.height = img.size
		#Hacky and lazy.
		if img.width % 8 != 0:
			newwidth = img.width + (8 - (img.width % 8))
			newimg = Image.new(img.mode, (newwidth, img.height))
			newimg.paste(img, (0, 0))
			img = newimg
		if 'format' in opts.keys():
			if 'ETC1' in opts['format'].upper():
				print('ETC1 not supported. Defaults to RGBA8')
				opts['format'] = 'RGBA8'
			self.format = FORMATS[opts['format'].upper()]
			self.strformat = opts['format'].upper()
		else:
			self.format = RGBA8
		if 'swizzle' not in opts.keys():
			opts['swizzle'] = '0'
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
		return self.pack(BFLIM_IMAG_HDR_STRUCT, b'imag', 0x10, img.width, img.height, ALIGNMENT, self.format, swizzle, datalen)
	
	def swizzle(self, img, swizzle):
		if self.verbose and swizzle != '0':
			print('Swizzling image')
		swizzle = int(swizzle)
		if swizzle == 4:
			img = img.rotate(-90, expand=True)
			img = img.crop((0, 0, self.height, self.width))
		elif swizzle == 8:
			img = img.transpose(Image.FLIP_TOP_BOTTOM)
			img = img.rotate(-90, expand=True)
			img = img.crop((0, 0, self.height, self.width))
		return img
	
	def repack_data(self, img):
		if self.verbose:
			print('Packing pixel data')
		self.pxsize = PIXEL_SIZES[self.format]
		pixels = img.load()
		width, height = img.size
		datawidth = 1 << int(math.ceil(math.log(width, 2)))
		dataheight = 1 << int(math.ceil(math.log(height, 2)))
		tiles_x = math.ceil(datawidth / 8)
		tiles_y = math.ceil(dataheight / 8)
		if self.verbose:
			print('Packing %d x %d tiles of %dB each' % (tiles_x, tiles_y, int(64 * self.pxsize)))
		final = bytearray(datawidth * dataheight * self.pxsize)
		for ytile in range(tiles_y):
			for xtile in range(tiles_x):
				for ysub in range(2):
					for xsub in range(2):
						for ygroup in range(2):
							for xgroup in range(2):
								for ypix in range(2):
									for xpix in range(2):
										posy = (ytile * 8) + (ysub * 4) + (ygroup * 2) + ypix
										posx = (xtile * 8) + (xsub * 4) + (xgroup * 2) + xpix
										if posx >= width or posy >= height:
											#rgba = (0, 0, 0, 0)
											continue
										else:
											rgba = pixels[posx, posy]
										packed = self.pack_pixel(rgba)
										if self.byteorder == '<':
											packed = bytes(reversed(packed))
										finalx = xpix + (xgroup * 4) + (xsub * 16) + (xtile * 64)
										finaly = (ypix * 2) + (ygroup * 8) + (ysub * 32) + (ytile * width * 8)
										pos = (finalx + finaly) * self.pxsize
										final[pos:pos + self.pxsize] = packed
		return final

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
		elif self.format == RGB8:
			return self.pack('3B', r, g, b)
		elif self.format == RGB565:
			r = math.ceil(r / 8.225806451612904) << 11
			g = math.ceil(g / 4.0476190476190474) << 5
			b = math.ceil(b / 8.225806451612904)
			return self.pack('H', r + g + b)
		elif self.format == RGBA5551:
			a = int(a != 0)
			r = math.ceil(r // 8.225806451612904) << 11
			g = math.ceil(g // 8.225806451612904) << 6
			b = math.ceil(b // 8.225806451612904) << 1
			return self.pack('H', r + g + b + a)
		elif self.format == RGBA4:
			r = (r // 0x11) << 12
			g = (g // 0x11) << 8
			b = (b // 0x11) << 4
			a = (a // 0x11)
			return self.pack('H', r + g + b + a)
		elif self.format == RGBA8:
			return self.pack('4B', r, g, b, a)
		else:
			error('Unsupported format %s' % self.strformat, 105)
