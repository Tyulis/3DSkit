# -*- coding:utf-8 -*-
import os
from util import error, ENDIANS
from util.utils import byterepr
from util.filesystem import *
import util.rawutil as rawutil
from unpack._formats import get_ext

NARC_HEADER_STRUCT = '4s2HI2H'
NARC_HEADER_NAMES = 'magic, bom, unknown, filelen, headerlen, sectioncount'
NARC_FATB_SECTION = '4s2I /p1[2I]'
NARC_FATB_NAMES = 'magic, sectionlen, entrycount, entries'
NARC_FNTB_SECTION = '4sI I2H'
NARC_FNTB_NAMES = 'magic, sectionlen, startoffset, firstfilepos, dircount'

class extractNARC (rawutil.TypeReader):
	def __init__(self, filename, file, verbose, opts={}):
		self.outdir = make_outdir(filename)
		self.verbose = verbose
		self.file = file
		self.readheader()
		self.readFATB()
		self.readFNTB()
	
	def readheader(self):
		self.byteorder = ENDIANS[rawutil.unpack_from('<H', self.file, 4)[0]]
		header = self.unpack_from(NARC_HEADER_STRUCT, self.file, 0, NARC_HEADER_NAMES)
		if header.magic != b'NARC':
			error.InvalidMagicError('Invalid NARC magic: %s' % byterepr(header.magic))

	def readFATB(self):
		fatb = self.unpack_from(NARC_FATB_SECTION, self.file, None, NARC_FATB_NAMES)
		if fatb.magic != b'BTAF':
			error.InvalidMagicError('Invalid FATB magic: %s' % byterepr(fatb.magic))
		self.entrycount = fatb.entrycount
		self.entries = fatb.entries

	def readFNTB(self):
		offset = self.file.tell()
		fntb = self.unpack_from(NARC_FNTB_SECTION, self.file, None, NARC_FNTB_NAMES)
		if fntb.magic != b'BTNF':
			error.InvalidMagicError('Invalid FNTB magic: %s' % byterepr(fntb.magic))
		if fntb.sectionlen == 0x10:  #No names...
			self.has_names = False
		else:
			self.has_names = True
			self.names = []
			for i in enumerate(self.entries):
			    file_info = self.unpack_from('B /p1s', self.file, None, 'length, name')
			    self.names.append(file_info.name.decode())
		self.fimgoffset = offset + fntb.sectionlen

	def extract(self):
		for i, entry in enumerate(self.entries):
			self.file.seek(self.fimgoffset + 8 + entry[0])
			filedata = self.file.read(entry[1] - entry[0])
			if self.has_names:
				file = open(self.outdir + self.names[i], 'wb')
				print('  Generating %s' % self.names[i])
			else:
				file = open(self.outdir + str(i) + get_ext(filedata), 'wb')
			file.write(filedata)
			file.close()
