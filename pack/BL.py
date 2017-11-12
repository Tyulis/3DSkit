# -*- coding:utf-8 -*-
import util.rawutil as rawutil
from util import error
from util.funcops import ClsFunc
from util.fileops import *

BL_TABLE_STRUCT = '2sH /1[I] 128a'


class packBL (ClsFunc, rawutil.TypeWriter):
	def main(self, filenames, outname, endian, verbose, opts={}):
		self.byteorder = endian
		self.verbose = verbose
		data, offsets = self.repack_files(filenames)
		header = self.repack_headers(offsets)
		final = header + data
		bwrite(final, outname)
	
	def repack_files(self, filenames):
		contents = [b''] * len(filenames)
		for name in filenames:
			try:
				num = int(name)
			except ValueError:
				error('File name %s does not have the right format. It should be like 003.xxx', 205)
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
		header = self.pack(BL_TABLE_STRUCT, b'BL', len(offsets), offsets)
		return header
