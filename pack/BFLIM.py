# -*- coding:utf-8 -*-
from util.funcops import ClsFunc
from util.rawutil import TypeWriter
from PIL import Image

ALIGNMENT = 0x80

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
		filename = filenames[0]
		img = Image.open(filename)
		if 'format' in opts.keys():
			self.format = FORMATS[opts['format'].upper()]
		else:
			self.format = RGBA8
		img = self.swizzle(img, opts['swizzle'])
		pixels = img.load()
		data = self.repack_data(pixels)
		flimhdr = self.repackFLIMheader(img)
		imaghdr = self.repackIMAGheader(img)
		final = data
		final += self.align(final, ALIGNMENT)
		final += flimhdr + imaghdr
		bwrite(final, outname)
	
	def swizzle(self, img, swizzle):
		swizzle = int(swizzle)
		if swizzle == 4:
			img = img.rotate(-90)
		elif swizzle == 8:
			img = img.transpose(Image.FLIP_LEFT_RIGHT)
			img = img.rotate(-90)
		return img
