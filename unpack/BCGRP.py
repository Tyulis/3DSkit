# -*- coding:utf-8 -*-
import os
from util import error, ENDIANS
from util.utils import byterepr
from util.filesystem import *
import util.rawutil as rawutil
from unpack._formats import get_ext

class extractBCGRP (rawutil.TypeReader):
	def __init__(self, filename, data, verbose, opts={}):
		self.outdir = make_outdir(filename)
		self.verbose = verbose
		self.read_header(data)
		if 'INFO' in self.secinfo:
			self.readINFO(data, self.secinfo['INFO'][0], self.secinfo['INFO'][1])
		if 'INFX' in self.secinfo:
			self.readINFX(data, self.secinfo['INFX'][0], self.secinfo['INFX'][1])
		self.data = data
	
	def read_header(self, data):
		magic, bom = rawutil.unpack_from('>4sH', data, 0)
		self.byteorder = ENDIANS[bom]
		if magic != b'CGRP':
			error.InvalidMagicError('Invalid magic %s, expected CGRP' % byterepr(magic))
		magic, bom, headerlen, self.version, filesize, self.seccount = self.unpack_from('4s2H3I', data, 0)
		self.secinfo = {}
		for i in range(self.seccount):
			id, pad, offset, size = self.unpack_from('2H2I', data)
			if id == 0x7800:
				self.secinfo['INFO'] = (offset, size)
			elif id == 0x7801:
				self.secinfo['FILE'] = (offset, size)
			elif id == 0x7802:
				self.secinfo['INFX'] = (offset, size)
		# Then 8 Bytes of padding
	
	def readINFO(self, data, start, size):
		data.seek(start)
		magic, size = self.unpack_from('4sI', data)
		if magic != b'INFO':
			error.InvalidMagicError('Invalid INFO magic (got %s)' % byterepr(magic))
		baseoffset = data.tell()
		entrycount, table = self.unpack_from('I/p1[2HI]', data)
		if self.verbose:
			print('Number of files: %d' % entrycount)
		self.fileinfo = []
		for id, pad, offset in table:
			if id == 0x7900:  # File reference offset
				id, pad, pos, size, unknown = self.unpack_from('2H3I', data, baseoffset + offset + 4)
				if id == 0x1f00:  # File reference
					self.fileinfo.append({'pos': pos, 'size': size, 'unknown': unknown})
				else:
					error.UnsupportedValueWarning('Unknown reference type %04X in INFO section' % id)
	
	def readINFX(self, data, start, size):
		data.seek(start)
		magic, size = self.unpack_from('4sI', data)
		if magic != b'INFX':
			error.InvalidMagicError('Invalid INFO magic (got %s)' % byterepr(magic))
		baseoffset = data.tell()
		entrycount, table = self.unpack_from('I/p1[2HI]', data)
		for id, pad, offset in table:
			if id == 0x7901:  # ???
				values = self.unpack_from('2I', data, baseoffset + offset)
				# What's that?

	def extract(self):
		data = self.data
		# Reading FILE section
		data.seek(self.secinfo['FILE'][0])
		magic, size = self.unpack_from('4sI', data)
		if magic != b'FILE':
			error.InvalidMagicError('Invalid FILE magic (got %s)' % byterepr(magic))
		baseoffset = data.tell()
		for i, info in enumerate(self.fileinfo):
			start = baseoffset + info['pos']
			data.seek(start)
			filedata = data.read(info['size'])
			ext = get_ext(filedata)
			filename = path(self.outdir, '%d%s' % (i, ext))
			with open(filename, 'wb') as out:
				out.write(filedata)
			