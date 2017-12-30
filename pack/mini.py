# -*- coding:utf-8 -*-
import util.rawutil as rawutil
from util import error
from util.utils import ClsFunc
from util.filesystem import *

MINI_TABLE_STRUCT = '2sH /1[I] 128a'


class packmini (ClsFunc, rawutil.TypeWriter):
	def main(self, filenames, outname, endian, verbose, opts={}):
		self.byteorder = endian
		self.verbose = verbose
		self.magic = b'BL'
		if 'magic' in opts.keys():
			if len(opts['magic']) != 2:
				error.InvalidOptionValueError('Mini file magic should be 2 characters length')
			self.magic = opts['magic'].encode('ascii')
		data, offsets = self.repack_files(filenames)
		header = self.repack_headers(offsets)
		final = header + data
		bwrite(final, outname)
	
	def repack_files(self, filenames):
		contents = [b''] * len(filenames)
		name = name.split('.')[0]
		for name in filenames:
			try:
				num = int(name)
			except ValueError:
				error.InvalidInputError('File name %s does not have the right format. It should be like <number>.<extension>')
			content = bread(filenames)
			contents[num] = content
		final = b''
		offsets = [0]  #Header length is added later
		for content in contents:
			final += content
			offsets.append(len(final))
		return final, offsets
	
	def repack_header(self, offsets):
		headerlen = 4 + len(offsets) * 4
		headerlen += 0x80 - (headerlen % 0x80)
		offsets = [[offset + headerlen] for offset in offsets]
		header = self.pack(MINI_TABLE_STRUCT, self.magic, len(offsets), offsets)
		return header
