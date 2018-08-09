# -*- coding:utf-8 -*-
import os
from util import error, ENDIANS
from util.utils import byterepr, switch, SWITCH_DEFAULT, ClsFunc
from util.filesystem import *
from util.txtree import dump
import util.rawutil as rawutil


class extractMSBT (rawutil.TypeReader, ClsFunc):
	def main(self, filename, data, verbose, opts={}):
		self.outfile = make_outfile(filename, 'txt')
		self.verbose = verbose
		if 'showescapes' in opts:
			self.escapes = True if opts['showescapes'].strip().lower() == 'true' else False
		else:
			self.escapes = True
		self.read_header(data)
		self.strnum = None
		for i in range(self.sectionnum):
			magic = data.read(4)
			data.seek(-4, 1)
			switch(magic, (data, magic), {
				b'LBL1': self.readLBL1,
				b'TXT2': self.readTXT2,
				b'ATR1': self.readATR1,
				b'ATO1': self.readATO1,
				b'NLI1': self.readNLI1,
				b'TSY1': self.readTSY1,
				SWITCH_DEFAULT: self.unknown_section,
			})
		dic = [None for i in range(self.strnum)]
		for names in self.names:
			'''group = {}
			for id, name in names:
				group[name] = self.strings[id + 1]
			dic.append(group)'''
			for id, name in names:
				dic[id] = {name: self.strings[id]}
		final = {'Strings': dic, 'Number of strings': self.strnum}
		out = dump(final)
		write(out, self.outfile)
		
	
	def read_header(self, data):
		magic = data.read(8)
		if magic != b'MsgStdBn':
			error.InvalidMagicError('Invalid magic %s, expected MsgStdBn' % byterepr(magic))
		self.byteorder = ENDIANS[rawutil.unpack_from('>H', data)[0]]
		magic, bom, unk1, self.encoding, unk2, self.sectionnum, unk3, self.filesize, unk4 = self.unpack_from('8s2H2B2HI10s', data, 0)
	
	def readLBL1(self, data, magic):
		magic, size, unknown = self.unpack_from('4sI8s', data)
		baseoffset = data.tell()
		if magic != b'LBL1':
			error.InvalidMagicError('Invalid LBL1 magic (%s)' % byterepr(magic))
		entrynum = self.unpack_from('I', data)[0]
		self.names = []
		for i in range(entrynum):
			count, offset = self.unpack_from('2I', data)
			names = []
			pos = data.tell()
			data.seek(offset + baseoffset)
			for n in range(count):
				length, name, id = self.unpack_from('B/p1sI', data)
				names.append((id, name.decode('ascii')))
			self.names.append(names)
			data.seek(pos)
		endpos = baseoffset + size
		endpos += 0x10 - (endpos % 0x10 or 0x10)
		data.seek(endpos)
	
	def readTXT2(self, data, magic):
		magic, size, unknown = self.unpack_from('4sI8s', data)
		baseoffset = data.tell()
		if magic != b'TXT2':
			error.InvalidMagicError('Invalid TXT2 magic (%s)' % byterepr(magic))
		entrynum, offsets = self.unpack_from('I/p1(I)', data)
		self.strings = []
		encoding = 'utf-16-%s' % ('le' if self.byteorder == '<' else 'be')
		for i in range(entrynum):
			start = offsets[i]
			if i < entrynum - 1:
				length = offsets[i + 1] - start
			else:
				length = size - start
			data.seek(baseoffset + start)
			strdata = data.read(length)
			pos = 0
			string = ''
			while pos < length:
				codepoint = strdata[pos: pos + 2]
				if codepoint == b'\x00\x00':
					self.strings.append(string)
					break
					string = ''
					pos += 2
				else:
					character = codepoint.decode(encoding)
					if ord(character) < 0x20 and character not in '\n\t':
						if self.escapes:
							character = '\\u%04X' % ord(character)
						else:
							character = ''
					string += character
					pos += 2
		endpos = baseoffset + size
		endpos += 0x10 - (endpos % 0x10 or 0x10)
		data.seek(endpos)
		
	def readATR1(self, data, magic):
		magic, size, unknown = self.unpack_from('4sI8s', data)
		baseoffset = data.tell()
		if magic != b'ATR1':
			error.InvalidMagicError('Invalid ATR1 magic (%s)' % byterepr(magic))
		self.strnum = self.unpack_from('I', data)[0]
		endpos = baseoffset + size
		endpos += 0x10 - (endpos % 0x10 or 0x10)
		data.seek(endpos)
	
	def readATO1(self, data, magic):
		magic, size, unknown = self.unpack_from('4sI8s', data)
		baseoffset = data.tell()
		if magic != b'ATO1':
			error.InvalidMagicError('Invalid ATO1 magic (%s)' % byterepr(magic))
		endpos = baseoffset + size
		endpos += 0x10 - (endpos % 0x10 or 0x10)
		data.seek(endpos)
		error.NotImplementedWarning('Support for section ATO1 is not implemented yet')
	
	def readTSY1(self, data, magic):
		magic, size, unknown = self.unpack_from('4sI8s', data)
		baseoffset = data.tell()
		if magic != b'TSY1':
			error.InvalidMagicError('Invalid TSY1 magic (%s)' % byterepr(magic))
		endpos = baseoffset + size
		endpos += 0x10 - (endpos % 0x10 or 0x10)
		data.seek(endpos)
		error.NotImplementedWarning('Support for section TSY1 is not implemented yet')
	
	def readNLI1(self, data, magic):
		magic, size, unknown = self.unpack_from('4sI8s', data)
		baseoffset = data.tell()
		if magic != b'NLI1':
			error.InvalidMagicError('Invalid NLI1 magic (%s)' % byterepr(magic))
		endpos = baseoffset + size
		endpos += 0x10 - (endpos % 0x10 or 0x10)
		data.seek(endpos)
		error.NotImplementedWarning('Support for section NLI1 is not implemented yet')
	
	def unknown_section(self, data, magic):
		error.InvalidSectionError('Unknown section %s' % byterepr(magic))