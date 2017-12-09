# -*- coding:utf-8 -*-
from util import error
from util.fileops import *
import util.rawutil as rawutil
from hashlib import sha256

EXEFS_HEADER_STRUCT = '10[8s2I] 32s 10[32s]'


class FileEntry (object):
	def __init__(self, hdr, hash):
		self.name = hdr[0].rstrip(b'\x00').decode('ascii')
		self.offset = hdr[1]
		self.size = hdr[2]
		self.hash = hash[0]


class extractExeFS (rawutil.TypeReader):
	def __init__(self, filename, data, verbose, opts={}):
		self.outdir = make_outdir(filename)
		self.verbose = verbose
		self.dochecks = False
		if 'dochecks' in opts.keys():
			self.dochecks = True if opts['dochecks'].lower() == 'true' else False
		self.byteorder = '<'
		self.readheader(data[:0x200])
		self.data = data[0x200:]
	
	def readheader(self, raw):
		hdrs, reserved, hashes = self.unpack_from(EXEFS_HEADER_STRUCT, raw)
		self.files = [FileEntry(hdrs[i], hashes[-(i + 1)]) for i in range(10) if hdrs[i][0] != bytes(8)]

	def extract(self):
		for file in self.files:
			content = self.data[file.offset: file.offset + file.size]
			if self.dochecks:
				if sha256(content).digest() != file.hash:
					paf
					error.HashMismatchError('File %s hash mismatch' % file.name)
			bwrite(content, self.outdir + file.name)
