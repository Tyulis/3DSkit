# -*- coding:utf-8 -*-
import os
import math
from util.utils import ClsFunc
import util.rawutil as rawutil
from util import error
import py3DSkit as libkit
import numpy as np
from PIL import Image

FILE_HEADER_STRUCT = rawutil.Struct(
	"<4sI8s 2I I 12x n",
	("magic", "version", "magic_texture", "fullsize", "delimiter", "datasize", "filename")
)

DATA_HEADER_STRUCT = rawutil.Struct(
	"<2I 2H 2H",
	("unk60", "unk64", "width", "height", "format", "swizzle")
)

DATA_FORMATS = {
	"RGB565": 0x02,
	"RGB8": 0x03,
	"RGBA8": 0x04,
	"RGBA4": 0x16,
	"RGBA5551": 0x17,
	"LA8": 0x23,
	"HILO8": 0x24,
	"L8": 0x25,
	"A8": 0x26,
	"LA4": 0x27,
	"L4": 0x28,
	"A4": 0x29,
	"ETC1": 0x2A,
	"ETC1A4": 0x2B,
}

PIXEL_SIZES = {
	"RGB565": 2,
	"RGB8": 3,
	"RGBA8": 4,
	"RGBA4": 2,
	"RGBA5551": 2,
	"LA8": 2,
	"HILO8": 2,
	"L8": 1,
	"A8": 1,
	"LA4": 1,
	"L4": 0.5,
	"A4": 0.5,
	"ETC1": 0.5,
	"ETC1A4": 1,
}

DATA_HEADER_OFFSET = 0x60
DATA_OFFSET = 0x80

MAGIC = b"\x13\x12\x04\x15"
VERSION = 0x00000001

class packPkmSMTexture (ClsFunc):
	def main(self, filenames, outname, endian, verbose, opts={}):
		self.byteorder = endian
		self.verbose = verbose
		filename = filenames[0]
		try:
			img = Image.open(filename)
		except:
			error.InvalidInputError('The given input file is not an image')
		img = img.convert('RGBA')
		self.width, self.height = img.size

		if 'format' in opts.keys():
			if 'ETC1' in opts['format'].upper():
				error.UnsupportedDataFormatWarning('ETC1 is not supported, packing as RGBA8')
				opts['format'] = 'RGBA8'
			self.format = opts['format'].upper()
		else:
			self.format = "RGBA8"
		if 'swizzle' not in opts.keys():
			opts['swizzle'] = '1'
		if 'name' not in opts.keys():
			opts["name"] = os.path.basename(filename).rpartition(".")[0] + ".tga"
		img = self.swizzle(img, opts['swizzle'])

		outfile = open(outname, "wb")
		self.repackData(img, outfile)
		self.repackDataHeader(img, outfile, int(opts["swizzle"]))
		self.repackFileHeader(img, outfile, opts["name"])

	def swizzle(self, img, swizzle):
		if self.verbose and swizzle != '1':
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

	def repackData(self, image, outfile):
		format = libkit.getTextureFormatId(self.format)
		if format == -1:
			error.UnsupportedDataFormatError('Texture format %s is not supported yet' % self.format)
		indata = np.ascontiguousarray(np.fromstring(image.tobytes(), dtype=np.uint8))
		outdata = np.ascontiguousarray(np.zeros(image.width * image.height * PIXEL_SIZES[self.format], dtype=np.uint8))
		libkit.packTexture(indata, outdata, image.width, image.height, format, -1, True)
		outfile.seek(DATA_OFFSET)
		outfile.write(outdata.tostring())
		self.filesize = outfile.tell()

	def repackDataHeader(self, image, outfile, swizzle):
		outfile.seek(DATA_HEADER_OFFSET)
		DATA_HEADER_STRUCT.pack_file(outfile, 0, 0, image.width, image.height, DATA_FORMATS[self.format], swizzle)
		outfile.write(b"\xFF" * 16)

	def repackFileHeader(self, image, outfile, filename):
		outfile.seek(0)
		FILE_HEADER_STRUCT.pack_file(outfile, MAGIC, VERSION, b"texture\x00", self.filesize - 0x18, 0xFFFFFFFF, self.filesize - 0x80, filename)
