# -*- coding:utf-8 -*-
import os
import json
import numpy as np
from PIL import Image
from util import error, ENDIANS, libkit
from util.utils import byterepr, ClsFunc
from util.filesystem import *
import util.rawutil as rawutil

DIMENSIONS = {
	0: '1D',
	1: '2D',
	2: '3D',
	3: 'Cube',
	4: '1D array',
	5: '2D array',
	6: '2D multi-sample',
	7: '2D multi-sample array',
	8: 'Cube array',
}

CHANNEL_SOURCES = {
	0: 'Zero', 1: 'One', 2: 'Red',
	3: 'Green', 4: 'Blue', 5: 'Alpha',
}

PIXEL_FORMATS = {
	0x02 : 'L8', 0x07 : 'RGB565', 0x09 : 'RG8', 0x0A : 'L16',
	0x0B : 'RGBA8', 0x0F : 'R11G11B10', 0x14 : 'L32', 0x1A : 'BC1',
	0x1B : 'BC2', 0x1C : 'BC3', 0x1D : 'BC4', 0x1E : 'BC5',
	0x1F : 'BC6H', 0x20 : 'BC7', 0x2D : 'ASTC4x4', 0x2E : 'ASTC5x4',
	0x2F : 'ASTC5x5', 0x30 : 'ASTC6x5', 0x31 : 'ASTC6x6', 0x32 : 'ASTC8x5',
	0x33 : 'ASTC8x6', 0x34 : 'ASTC8x8', 0x35 : 'ASTC10x5', 0x36 : 'ASTC10x6',
	0x37 : 'ASTC10x8', 0x38 : 'ASTC10x10', 0x39 : 'ASTC12x10', 0x3A : 'ASTC12x12',
}
VALUE_FORMATS = {
	0x01 : 'UNorm', 0x02 : 'SNorm', 0x03 : 'UInt', 0x04 : 'SInt',
	0x05 : 'Single', 0x06 : 'SRGB', 0x0A : 'UHalf',
}

PIXEL_SIZES = {
	'L8': 1, 'RGB565': 2, 'RG8': 2, 'L16': 2,
	'RGBA8': 4, 'R11G11B10': 4, 'L32': 4,
}

class extractBNTX (rawutil.TypeReader, ClsFunc):
	def main(self, filename, data, verbose, opts={}):
		self.verbose = verbose
		self.filebase = os.path.splitext(filename)[0]
		self.meta = {}
		self.read_header(data)
		self.read_container(data)
		self.read_textable(data)
		texmeta = []
		self.texturepos = self.brtd_offset + 0x10
		for brtioffset in self.brtioffsets:
			texmeta.append(self.read_texture(data, brtioffset))
		self.meta['textures'] = texmeta
		with open(self.filebase + '_meta.json', 'w') as f:
			json.dump(self.meta, f, indent=4)

	def read_header(self, data):
		startpos = data.tell()
		node = {}
		magic, bom = rawutil.unpack_from('>8s4xH', data)
		self.byteorder = ENDIANS[bom]
		data.seek(startpos)
		magic, self.version, bom, alignment_exponent, target_adress_size = self.unpack_from('8sIH2B', data)
		filename_offset, reloc_flag, self.firstsection_offset, self.reloc_offset, filesize = self.unpack_from('I2H2I', data)
		endpos = data.tell()
		if magic != b'BNTX\x00\x00\x00\x00':
			error.InvalidMagicError('Invalid BNTX magic, found %s' % byterepr(magic))
		self.alignment = 1 << alignment_exponent
		node['version'] = self.version
		node['alignment_exponent'] = alignment_exponent
		node['target_adress_size'] = target_adress_size
		node['relocation_flag'] = reloc_flag
		filename = self.unpack_from('n', data, filename_offset)[0].decode('utf-8')
		node['filename'] = filename
		self.meta['BNTX'] = node
		data.seek(endpos)
		if self.verbose:
			print('File version : %s.%s.%s.%s' % (self.version >> 24, (self.version >> 16) & 0xFF, (self.version >> 8) & 0xFF, (self.version & 0xFF)))

	def read_container(self, data):
		startpos = data.tell()
		node = {}
		platform, self.texnum, self.textable_offset, self.brtd_offset, self.dic_offset, self.texmempool_offset, self.currentmempool_pointer, self.basemempool_offset, unknown = self.unpack_from('4sI5Q2I', data)
		node['platform'] = platform.decode('utf-8').strip()
		node['unknown'] = unknown
		self.meta['container'] = node

	def read_textable(self, data):
		data.seek(self.textable_offset)
		self.brtioffsets = self.unpack_from('%dQ' % self.texnum, data)

	def read_texture(self, data, startpos):
		node = {}
		data.seek(startpos)
		magic, nextoffset, seclen, seclen2, unk1, tilemode, swizzle_value, mipmapnum, multisampnum, unk2 = self.unpack_from('4s3I6H', data)
		format, gpuaccesstype, width, height, depth, arraylength, blockheight_exponent, unk3, unk4 = self.unpack_from('8I20s', data)
		mipmapdata_size, texdata_alignment, redsource, greensource, bluesource, alphasource, dimension = self.unpack_from('2I5B3x', data)
		texname_offset, texcontainer_offset, textable_offset, userdata_offset, texpointer, texview_pointer, descriptorslot_offset, userdata_dict_offset = self.unpack_from('8q', data)
		endpos = data.tell()
		if magic != b'BRTI':
			error.InvalidMagicError('Invalid BRTI magic, found %s' % byterepr(magic))
		if dimension not in (1, 5):
			error.UnsupportedSettingError('Only 2D textures are currently supported (found %s)' % DIMENSIONS[dimension])
		texname = self.unpack_from('H/p1s', data, texname_offset)[1].decode('utf-8')
		pixelformat = (format >> 8) & 0xFF
		valueformat = (format & 0xFF)
		node['tilemode'] = tilemode
		node['swizzle_value'] = swizzle_value
		node['mipmap_number'] = mipmapnum
		node['multi_samples_num'] = multisampnum
		node['pixel_format'] = PIXEL_FORMATS[pixelformat]
		node['value_format'] = VALUE_FORMATS[valueformat]
		node['gpu_access_type'] = gpuaccesstype
		node['array_length'] = arraylength
		node['block_height_exponent'] = blockheight_exponent
		node['texture_data_alignment'] = texdata_alignment
		node['red_source'] = redsource
		node['green_source'] = greensource
		node['blue_source'] = bluesource
		node['alpha_source'] = alphasource
		node['dimension'] = DIMENSIONS[dimension]
		node['texture_name'] = texname
		node['unk1'] = unk1
		node['unk2'] = unk2
		node['unk3'] = unk3
		node['unk4'] = unk4.hex()
		for i in range(arraylength):
			print('Extracting texture %s, element %d' % (texname, i))
			data.seek(self.texturepos)
			format = libkit.getTextureFormatId(PIXEL_FORMATS[pixelformat])
			if format == -1:
				error.UnsupportedDataFormatError('Pixel format %s is not supported yet' % PIXEL_FORMATS[pixelformat])
			datasize = self.data_size(PIXEL_FORMATS[pixelformat], width, height)
			indata = np.ascontiguousarray(np.fromstring(data.read(datasize), dtype=np.uint8))
			out = np.ascontiguousarray(np.zeros(width * height * 4, dtype=np.uint8))
			libkit.extractTiledTexture(indata, out, width, height, format, swizzle_value if tilemode == 0 else -1, self.byteorder == '<')
			img = Image.frombytes('RGBA', (width, height), out.tostring())
			outname = '%s_%s_el%d.png' % (self.filebase, texname, i)
			img.save(outname, 'PNG')
			self.texturepos += datasize + (texdata_alignment - (datasize % texdata_alignment or texdata_alignment))
		return node

	def data_size(self, format, width, height):
		if format in PIXEL_SIZES:
			return PIXEL_SIZES[format] * width * height
		else:
			error.UnsupportedDataFormatError('Pixel format %s is not supported yet' % format)
