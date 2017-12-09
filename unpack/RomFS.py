# -*- coding:utf-8 -*-
import os
from util import error
from util.funcops import byterepr
from util.fileops import *
import util.rawutil as rawutil

ROMFS_HEADER_STRUCT = '4s2I 2Q2I 2Q2I 2Q2I 2I'
ROMFS_LEVEL3_HEADER_STRUCT = '<10I'

class extractRomFS (rawutil.TypeReader):
	def __init__(self, filename, data, opts={}):
		self.outdir = make_outdir(filename)
		self.byteorder = '<'
		self.readheader(data[:0x5c])
		self.read_level3(data[self.level3_offset: self.level3_offset + self.level3_hashdata_size])
	
	def readheader(self, raw):
		data = self.unpack(ROMFS_HEADER_STRUCT, raw)
		magic = data[0]
		if magic != b'IVFC':
			error.InvalidMagicError('Invalid magic %s, expected IVFC' % byterepr(magic))
		if data[1] != 0x10000:
			error.InvalidMagicError('Invalid magic 0x%08x, expected 0x00010000' % data[1])
		self.masterhash_size = data[2]
		self.level1_offset = data[3]
		self.level1_hashdata_size = data[4]
		self.level1_blocksize = 2 ** data[5]
		reserved = data[6]
		self.level2_offset = data[7]
		self.level2_hashdata_size = data[8]
		self.level2_blocksize = 2 ** data[9]
		reserved = data[10]
		self.level3_offset = data[11]
		self.level3_hashdata_size = data[12]
		self.level3_blocksize = 2 ** data[13]
		reserved = data[14:16]
		self.optionalinfo_size = data[16]
	
	def read_level3(self, raw):
		data = self.unpack_from(ROMFS_LEVEL3_HEADER_STRUCT, raw)
		del raw
		headerlen = data[0]
		dirs_hashtable_offset = data[1]
		dirs_hashtable_length = data[2]
		dirs_metatable_offset = data[3]
		dirs_metatable_length = data[4]
		files_hashtable_offset = data[5]
		files_hashtable_length = data[6]
		files_metatable_offset = data[7]
		files_metatable_length = data[8]
		filedata_offset = data[9]
		pag

	def extract(self):
		#Code to really extract files
		pass
