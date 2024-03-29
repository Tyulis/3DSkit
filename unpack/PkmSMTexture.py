import os
import numpy as np
from collections import namedtuple
import util.rawutil as rawutil
from util.utils import ClsFunc, byterepr
from util import error, libkit
from PIL import Image


FILE_HEADER_STRUCT = "<4sI8s 2I I 12x n"
FILE_HEADER_NAMES = "magic version magic_texture fullsize delimiter datasize filename"

DATA_HEADER_STRUCT = "<2I 2H 2H"
DATA_HEADER_NAMES = "unk60 unk64 width height format swizzle"

DATA_FORMATS = {
	0x02: "RGB565",
	0x03: "RGB8",
	0x04: "RGBA8",
	0x16: "RGBA4",
	0x17: "RGBA5551",
	0x23: "LA8",
	0x24: "HILO8",
	0x25: "L8",
	0x26: "A8",
	0x27: "LA4",
	0x28: "L4",
	0x29: "A4",
	0x2A: "ETC1",
	0x2B: "ETC1A4",
}

DATA_HEADER_OFFSET = 0x60
DATA_OFFSET = 0x80

class extractPkmSMTexture (ClsFunc):
	def main(self, filename, data, verbose, opts={}):
		self.verbose = verbose
		self.filename = filename
		self.readHeader(data)
		self.readDataHeader(data)
		self.extract(data)

	def readHeader(self, data):
		self.fileheader = rawutil.unpack_from(FILE_HEADER_STRUCT, data, names=FILE_HEADER_NAMES)
		if self.fileheader.magic_texture != b"texture\x00":
			error.InvalidMagicError("Invalid magic %s, expected texture\x00" % byterepr(self.fileheader.magic_texture))
		self.outname = os.path.join(os.path.dirname(self.filename), self.fileheader.filename.decode("utf-8").rpartition(".")[0] + ".png")
		if self.verbose:
			print("Version : %08X" % self.fileheader.version)
		print("Origin file name : %s" % self.fileheader.filename.decode("utf-8"))

	def readDataHeader(self, data):
		data.seek(DATA_HEADER_OFFSET)
		self.dataheader = rawutil.unpack_from(DATA_HEADER_STRUCT, data, names=DATA_HEADER_NAMES)
		if self.dataheader.format not in DATA_FORMATS:
			error.UnknownDataFormatError("Unknown texture data format identifier 0x%04X" % self.dataheader.format)
		print("Color format : %s" % DATA_FORMATS[self.dataheader.format])
		print("Texture swizzling : %d" % self.dataheader.swizzle)

	def extract(self, data):
		format = libkit.getTextureFormatId(DATA_FORMATS[self.dataheader.format])
		if format == -1:
			error.UnsupportedDataFormatError('Unsupported texture format %s' % DATA_FORMATS[self.format])

		data.seek(DATA_OFFSET)
		indata = np.ascontiguousarray(np.fromstring(data.read(), dtype=np.uint8))
		out = np.ascontiguousarray(np.zeros(self.dataheader.width * self.dataheader.height * 4, dtype=np.uint8))
		libkit.extractTiledTexture(indata, out, self.dataheader.width, self.dataheader.height, format, -1, True)
		img = Image.frombytes('RGBA', (self.dataheader.width, self.dataheader.height), out.tostring())
		img = self.deswizzle(img)
		img.save(self.outname, 'PNG')

	def deswizzle(self, img):
		if self.verbose and self.dataheader.swizzle != 0:
			print('Deswizzling')
		if self.dataheader.swizzle == 4:
			img = img.transpose(Image.ROTATE_90)
		elif self.dataheader.swizzle == 8:
			img = img.transpose(Image.ROTATE_90)
			img = img.transpose(Image.FLIP_TOP_BOTTOM)
		return img
