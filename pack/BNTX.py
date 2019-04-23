# -*- coding:utf-8 -*-
import json
import math
import numpy as np
from PIL import Image
from util import error, BOMS, libkit
from util.utils import ClsFunc
from util.rawutil import TypeWriter
from util.filesystem import *

BNTX_VERSION = 0x00040000

PLATFORMS = {'NX': b'NX  ', 'GEN': b'GEN '}

DIMENSIONS = {
	'1D': 0,
	'2D': 1,
	'3D': 2,
	'Cube': 3,
	'1D array': 4,
	'2D array': 5,
	'2D multi-sample': 6,
	'2D multi-sample array': 7,
	'Cube array': 8,
}

CHANNEL_SOURCES = {
	'Zero': 0, 'One': 1, 'Red': 2,
	'Green': 3, 'Blue': 4, 'Alpha': 5,
}

PIXEL_FORMATS = {
	'L8': 0x02, 'RGB565': 0x07, 'RG8': 0x09, 'L16': 0x0A,
	'RGBA8': 0x0B, 'R11G11B10': 0x0F, 'L32': 0x14, 'BC1': 0x1A,
	'BC2': 0x1B, 'BC3': 0x1C, 'BC4': 0x1D, 'BC5': 0x1E,
	'BC6H': 0x1F, 'BC7': 0x20, 'ASTC4x4': 0x2D, 'ASTC5x4': 0x2E,
	'ASTC5x5': 0x2F, 'ASTC6x5': 0x30, 'ASTC6x6': 0x31, 'ASTC8x5': 0x32,
	'ASTC8x6': 0x33, 'ASTC8x8': 0x34, 'ASTC10x5': 0x35, 'ASTC10x6': 0x36,
	'ASTC10x8': 0x37, 'ASTC10x10': 0x38, 'ASTC12x10': 0x39, 'ASTC12x12': 0x3A,
}
VALUE_FORMATS = {
	'UNorm': 0x01, 'SNorm': 0x02, 'UInt': 0x03, 'SInt': 0x04,
	'Single': 0x05, 'SRGB': 0x06, 'UHalf': 0x0A,
}

PIXEL_SIZES = {
	'L8': 1, 'RGB565': 2, 'RG8': 2, 'L16': 2,
	'RGBA8': 4, 'R11G11B10': 4, 'L32': 4,
}

STR_OFFSET = 0x01A0

class packBNTX(ClsFunc, TypeWriter):
	def main(self, filenames, outname, endian, verbose, opts={}):
		self.byteorder = endian
		self.verbose = verbose
		inname = filenames[0]
		with open(inname, 'r') as metafile:
			self.meta = json.load(metafile)
		self.out = open(outname, 'wb+')
		self.strings = self.get_strings()
		self.pack_strings(self.out, STR_OFFSET)
		self.brtioffsets = []
		self.images = []
		for i, texture in enumerate(self.meta['textures']):
			offset, images = self.pack_brti(self.out, texture)
			self.brtioffsets.append(offset)
			self.images.extend(images)
			if i > 0:
				pos = self.out.tell()
				self.out.seek(self.brtioffsets[i - 1] + 4)
				size = offset - self.brtioffsets[i - 1]
				self.pack('2I', size, size, self.out)
				self.out.seek(pos)
		self.pack_textable(self.out, self.brtioffsets)
		self.pack_brtd(self.out, max(self.brtioffsets), self.meta['textures'], self.images)
		self.pack_reloc(self.out)
		self.filesize = self.out.tell()
		self.pack_header(self.out)
		self.out.close()

	def get_strings(self):
		strings = {self.meta['BNTX']['filename']}  # Eliminates doubles
		for texture in self.meta['textures']:
			name = texture['texture_name']
			strings.add(name)
		return tuple(strings)

	def string_offset(self, string):
		index = self.strings.index(string)
		position = sum([len(previous) + 3 for previous in self.strings[:index]])
		return self.strstart + position

	def pack_strings(self, out, offset):
		out.seek(offset)
		dic = bytes.fromhex(self.meta['dictionary'])
		#strlength = 24 + sum([len(string) + 2 for string in self.strings])
		self.stroffset = offset
		out.write(bytes(24))
		self.strstart = out.tell()
		self.pack('{Hn4a}', [(len(string), string) for string in self.strings], out)
		strlength = out.tell() - self.stroffset
		padding = (0x10 - (strlength % 0x10 or 0x10)) * b'\x00'
		out.write(padding)
		self.dic_offset = out.tell()
		out.write(dic)
		endpos = out.tell()
		sectionsize = out.tell() - self.stroffset
		out.seek(self.stroffset)
		self.pack('4s5I', b'_STR', sectionsize, sectionsize, 0, len(self.strings), 0, out)
		out.seek(endpos)

	def pack_brti(self, out, texinfo):
		startpos = out.tell()
		images = []
		tilemode = texinfo['tilemode']
		swizzle = texinfo['swizzle_value']
		self.pack('4s12x6H', b'BRTI', texinfo['unk1'], tilemode, swizzle, texinfo['mipmap_number'], texinfo['multi_samples_num'], texinfo['unk2'], out)
		pixelformat = PIXEL_FORMATS[texinfo['pixel_format']]
		valueformat = VALUE_FORMATS[texinfo['value_format']]
		format = (pixelformat << 8) | valueformat
		try:
			for element in texinfo['output']:
				images.append(Image.open(element))
		except IOError:
			error.FileNotFoundError('The file given by the JSON meta-data file "%s" is not found')
		width = images[0].width
		height = images[0].height
		depth = texinfo['depth']
		arraylength = texinfo['array_length']
		blockheight_exponent = texinfo['block_height_exponent']
		self.pack('8I20s4x', format, texinfo['gpu_access_type'], width, height, depth, arraylength, blockheight_exponent, texinfo['unk3'], bytes.fromhex(texinfo['unk4']), out)
		redsource = CHANNEL_SOURCES[texinfo['red_source']]
		greensource = CHANNEL_SOURCES[texinfo['green_source']]
		bluesource = CHANNEL_SOURCES[texinfo['blue_source']]
		alphasource = CHANNEL_SOURCES[texinfo['alpha_source']]
		dimension = DIMENSIONS[texinfo['dimension']]
		self.pack('I5B3x', texinfo['texture_data_alignment'], redsource, greensource, bluesource, alphasource, dimension, out)
		self.pack('8q', self.string_offset(texinfo['texture_name']), 0x20, startpos + 672, 0, startpos + 160, startpos + 416, 0, 0, out)
		return startpos, images

	def pack_textable(self, out, brtioffsets):
		self.textable_offset = self.stroffset - len(brtioffsets) * 8
		out.seek(self.textable_offset)
		self.pack('(%dQ)' % len(brtioffsets), brtioffsets, out)

	def pack_brtd(self, out, minposition, textures, images):
		self.brtd_offset = minposition + (0x1000 - (minposition % 0x1000 or 0x1000)) - 0x10
		out.seek(self.brtd_offset)
		self.pack('4sIQ', b'BRTD', 0, 0, out)
		texsizes = []
		texoffsets = []
		imgindex = 0
		for texindex, texture in enumerate(textures):
			startpos = out.tell()
			for element, outfile in enumerate(texture['output']):
				print('Packing texture %d, element %d' % (texindex, element))
				self.pack_texture(out, texture, images[imgindex])
				imgindex += 1
			texsizes.append(out.tell() - startpos)
			texoffsets.append(startpos)

		size = out.tell() - self.brtd_offset
		out.seek(self.brtd_offset + 8)
		self.pack('Q', size, out)

		self.out.seek(self.brtioffsets[-1] + 4)
		size = self.brtd_offset - self.brtioffsets[-1]
		self.pack('2I', size, size, self.out)

		self.headerend = 0
		for texsize, texoffset, brtioffset in zip(texsizes, texoffsets, self.brtioffsets):
			out.seek(brtioffset + 80)
			self.pack('I', texsize, out)
			out.seek(brtioffset + 672)
			self.pack('I', texoffset, out)
			alignedpos = out.tell() + (0x10 - (out.tell() % 0x10 or 0x10))
			if alignedpos > self.headerend:
				self.headerend = alignedpos

	def pack_texture(self, out, texinfo, image):
		format = libkit.getTextureFormatId(texinfo['pixel_format'])
		if format == -1:
			error.UnsupportedDataFormatError('Texture format %s is not supported yet' % texinfo['format'])
		indata = np.ascontiguousarray(np.fromstring(image.tobytes(), dtype=np.uint8))
		outdata = np.ascontiguousarray(np.zeros(self.texture_size(texinfo, image), dtype=np.uint8))
		libkit.packTexture(indata, outdata, image.width, image.height, format, texinfo['swizzle_value'] if texinfo['tilemode'] == 0 else -1, self.byteorder == '<')
		out.write(outdata.tostring())
		alignment = 0x200 - (outdata.shape[0] % 0x200 or 0x200)
		out.write(alignment * b'\x00')

	def texture_size(self, texinfo, image):
		padwidth = 2 ** math.ceil(math.log2(image.width))
		padheight = 2 ** math.ceil(math.log2(image.height))
		if texinfo['pixel_format'] in PIXEL_SIZES:
			return padheight * padwidth * PIXEL_SIZES[texinfo['pixel_format']]
		elif texinfo['pixel_format'] == 'BC4':
			return (padwidth * padheight) // 2
		else:
			error.UnsupportedDataFormatError('Texture format %s is not supported yet' % texinfo['format'])

	def pack_reloc(self, out):
		out.seek(0, 2)
		self.rltoffset = out.tell()
		self.pack('4s3I', b'_RLT', self.rltoffset, 2, 0, out)
		# TODO : check if it is always this
		self.pack('Q4I', 0, 0, self.headerend, 0, len(self.brtioffsets) + 3, out)
		self.pack('Q4I', 0, self.brtd_offset, self.rltoffset - self.brtd_offset, 4, 1, out)
		self.pack('IH2B', 0x28, 2, 1, 45, out)
		self.pack('IH2B', 0x38, 2, 2, 68, out)
		self.pack('IH2B', self.dic_offset + 0x10, 2, 1, 1, out)
		for offset in self.brtioffsets:
			self.pack('IH2B', offset + 96, 1, 3, 0, out)
		self.pack('IH2B', 0x30, 2, 1, 138, out)

	def pack_header(self, out):
		out.seek(0)
		self.pack('4s4xIHBB', b'BNTX', BNTX_VERSION, 0xFEFF, self.meta['BNTX']['alignment_exponent'], self.meta['BNTX']['target_adress_size'], out)
		self.pack('I2H2I', self.string_offset(self.meta['BNTX']['filename']) + 2, 0, self.stroffset, self.rltoffset, self.filesize, out)
		self.pack('4sI5Q2I', PLATFORMS[self.meta['container']['platform']], len(self.meta['textures']), self.textable_offset, self.brtd_offset, self.dic_offset, 0x58, 0, 0, self.meta['container']['unknown'], out)
