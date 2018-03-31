# -*- coding:utf-8 -*-
import util.rawutil as rawutil
from util import error
from util.utils import ClsFunc
from util.filesystem import *

MINI_TABLE_STRUCT = '2sH /1[I]'


class packmini (ClsFunc, rawutil.TypeWriter):
	def main(self, filenames, outname, endian, verbose, opts={}):
		self.byteorder = endian
		self.verbose = verbose
		self.magic = b'BL'
		if 'magic' in opts.keys():
			if len(opts['magic']) != 2:
				error.InvalidOptionValueError('Mini file magic should be 2 characters long')
			self.magic = opts['magic'].encode('ascii')
		data, offsets = self.repack_files(filenames)
		header = self.repack_header(offsets)
		final = header + data
		basedir()
		bwrite(final, outname)
	
	def repack_files(self, filenames):
		contents = [b''] * len(filenames)
		for name in filenames:
			stripped_name = name.split('.')[0]
			try:
				num = int(stripped_name)
			except ValueError:
				error.InvalidInputError('File name %s does not have the right format. It should be like <number>.<extension>')
			content = bread(name)
			contents[num] = content
		final = b''
		offsets = [0]  #Header length is added later
		for content in contents:
			final += content
			offsets.append(len(final))
		return final, offsets
	
	def repack_header(self, offsets):
		headerlen = 4 + len(offsets) * 4
		#headerlen += 0x80 - (headerlen % 0x80)
		offsets = [[offset + headerlen] for offset in offsets]
		header = self.pack("2sH %d[I]" % len(offsets), self.magic, len(offsets) - 1, offsets)  #-1 for the last one
		return header
