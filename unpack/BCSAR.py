# -*- coding:utf-8 -*-
import os
from util import error, ENDIANS
from util.utils import byterepr
from util.filesystem import *
import util.rawutil as rawutil

CSAR_HEADER = '4s2H2I2H /p2[2H2I]'
STRG_HEADER = '4sI 2[2HI]'
STRG_STR_OFFSETTABLE = 'I /p1[2H2I]'
INFO_HEADER = '4sI 8[2HI]'
INFO_SECTION = 'I/0[2I]'

# INFO tables
class INFOTable (rawutil.TypeReader):
	byteorder = None
	bcsar = None
	def __init__(self, offset):
		self.offset = offset


class TableEntry (object):
	def __init__(self, type=None, **kwargs):
		self.type = type
		self.__dict__.update(kwargs)
	
	def __repr__(self):
		return '%s entry: %s' % (self.type, ', '.join(['%s = %s' % (el, self.__dict__[el]) for el in self.__dict__]))


class AudioTable (INFOTable):
	id = 0x2100
	offsetid = 0x2200
	def extract(self, data, offsettbl):
		self.entries = []
		for offsetid, offset in offsettbl:
			fileid = self.unpack_from('I', data, offset)[0]
			id = self.unpack_from('H', data, offset + 12)[0]
			nameid = self.unpack_from('I', data, offset + 24)[0]
			name = self.bcsar.strg.strings[nameid]
			if id == 0x2201:  #External BCSTM
				entry = TableEntry('BCSTM', nameid=nameid, name=name, fileid=fileid)
			elif id == 0x2202:  #WSD
				entry = TableEntry('BCWSD', nameid=nameid, name=name, fileid=fileid)
			elif id == 0x2203:  #SEQ
				bank = self.unpack_from('H', data, offset + 92)
				entry = TableEntry('BCSEQ', nameid=nameid, name=name, fileid=fileid, bank=bank)
			self.entries.append(entry)
			

class BankTable (INFOTable):
	id = 0x2101
	offsetid = 0x2206
	def extract(self, data, offsettbl):
		self.entries = []
		for offsetid, offset in offsettbl:
			fileid, unk1, unk2, unk3, nameid, unk4 = self.unpack_from('6I', data, offset)
			name = self.bcsar.strg.strings[nameid]
			entry = TableEntry('BCBNK', fileid=fileid, nameid=nameid, name=name)
			self.entries.append(entry)

class PlayerTable (INFOTable):
	id = 0x2102
	def extract(self, data, offsettbl):
		pass

class WaveArcTable (INFOTable):
	id = 0x2103
	offsetid = 0x2207
	def extract(self, data, offsettbl):
		self.entries = []
		for i, (offsetid, offset) in enumerate(offsettbl):
			if i < len(offsettbl) - 1:
				datalen = offsettbl[i + 1][1] - offset
			else:
				datalen = self.info.grouptbl.offset - offset
			if datalen > 12:
				fileid, unk1, unk2, nameid = self.unpack_from('4I', data, offset)
				name = self.bcsar.strg.strings[nameid]
			else:
				fileid = self.unpack_from('3I', data, offset)[0]
				name = 'WAR_%08x' % i
				nameid = 0xffffffff
			entry = TableEntry('BCWAR', fileid=fileid, name=name, nameid=nameid)
			self.entries.append(entry)

class SetTable (INFOTable):  #possible error
	id = 0x2104
	offsetid = 0x2204
	def extract(self, data, offsettbl):
		self.entries = []
		for offsetid, offset in offsettbl:
			if offset == 0xffffffff:
				continue
			header = self.unpack_from('I', data, offset)
			if header == 0xffffffff:
				continue
			nameid, unk, fileid = self.unpack_from('3I', data, offset + 28)
			name = self.bcsar.strg.strings[nameid]
			entry = TableEntry('Set', fileid=fileid, nameid=nameid, name=name)
			self.entries.append(entry)

class GroupTable (INFOTable):
	id = 0x2105
	offsetid = 0x2208
	def extract(self, data, offsettbl):
		self.entries = []
		for offsetid, offset in offsettbl:
			fileid, unk, nameid = self.unpack_from('3I', data, offset)
			name = self.bcsar.strg.strings[nameid]
			entry = TableEntry('BCGRP', fileid=fileid, nameid=nameid, name=name)
			self.entries.append(entry)

class FileTable (INFOTable):
	id = 0x2106
	offsetid = 0x220a
	endid = 0x220b
	def extract(self, data, offsettbl):
		self.entries = []
		for offsetid, offset in offsettbl:
			id = self.unpack_from('2H2I', data, offset)[0]
			entry = TableEntry('File', internal=False, valid=True)
			if id == 0x220d:  #External
				entry.internal = False
				entry.path = self.unpack_from('n', data)[0]
			elif id == 0x220c:  #Internal
				entry.internal = True
				header, padding, offset, size = self.unpack_from('2H2I', data)
				if header != 0xffff and offset != 0xffffffff:
					entry.header = header
					entry.offset = offset
					entry.size = size
				else:
					entry.valid = False
			else:
				entry.valid = False
			self.entries.append(entry)


#BCSAR sections

class BCSARSection (rawutil.TypeReader):
	bcsar = None
	byteorder = None

class STRG (BCSARSection):
	id = 0x2000
	type = 'STRG'
	def __init__(self):
		self.offset = self.size = None
		self.strings = []
	
	def extract(self, data):
		header = self.unpack_from(STRG_HEADER, data, self.offset)
		if header[0] != b'STRG':
			error.InvalidMagicError('Invalid magic %s, expected STRG' % byterepr(header[0]))
		#sectionlen = header[1]
		for i, el in enumerate(header[2]):
			id, padding, offset = el
			if id == 0x2400:  #STRinGs table offset
				self.strtbl_offset = offset + 8 + (8 * i)
			elif id == 0x2401:  #Another offset table
				self.othertbl_offset = offset + 8 + (8 * i)
			else:
				error.InvalidSectionError('Invalid section ID in STRG 0x%04x' % id)
		strtbl_offsettbl = self.unpack_from(STRG_STR_OFFSETTABLE, data, self.strtbl_offset + self.offset)
		self.strings_count = strtbl_offsettbl[0]
		for entry in strtbl_offsettbl[1]:
			id, padding, offset, size = entry
			if id != 0x1f01:
				error.InvalidSectionError('Invalid entry ID in strings table offsets 0x%04x, expected 0x1f01' % id)
			else:
				self.strings.append(self.unpack_from('n', data, offset + self.strtbl_offset + 0x40)[0].decode('utf-8'))
		

class INFO (BCSARSection):
	id = 0x2001
	type = 'INFO'
	def __init__(self):
		self.offset = self.size = None
	
	def extract(self, data):
		header = self.unpack_from(INFO_HEADER, data, self.offset)
		if header[0] != b'INFO':
			error.InvalidMagicError('Invalid magic %s, expected INFO' % header[0])
		#sectionlen = header[1]
		for i, el in enumerate(header[2]):
			id, padding, offset = el
			if id == AudioTable.id:
				self.audiotbl = AudioTable(offset + self.offset + 8)
			elif id == BankTable.id:
				self.banktbl = BankTable(offset + self.offset + 8)
			elif id == PlayerTable.id:
				self.playertbl = PlayerTable(offset + self.offset + 8)
			elif id == WaveArcTable.id:
				self.warctbl = WaveArcTable(offset + self.offset + 8)
			elif id == SetTable.id:
				self.settbl = SetTable(offset + self.offset + 8)
			elif id == GroupTable.id:
				self.grouptbl = GroupTable(offset + self.offset + 8)
			elif id == FileTable.id:
				self.filetbl = FileTable(offset + self.offset + 8)
			elif id == FileTable.endid:
				self.filetbl.end = offset
		self.readTable(data, self.filetbl)
		self.readTable(data, self.warctbl)
		self.readTable(data, self.banktbl)
		self.readTable(data, self.audiotbl)
		self.readTable(data, self.settbl)
		self.readTable(data, self.grouptbl)
	
	def readOffsetTable(self, data, offset):
		entrycount, offsettbl = self.unpack_from(INFO_SECTION, data, offset)
		offsettbl = [(el[0], el[1] + offset) for el in offsettbl]
		return offsettbl
	
	def readTable(self, data, obj):
		offsettbl = self.readOffsetTable(data, obj.offset)
		obj.extract(data, offsettbl)
		

class FILE (BCSARSection):
	id = 0x2002
	type = 'FILE'
	def __init__(self):
		self.offset = self.size = None
	
	def extract(self, data):
		linkout = ''
		info = self.bcsar.info
		self.files = info.filetbl.entries
		linkout += self.processTable(data, info.warctbl)
		linkout += self.processTable(data, info.audiotbl)
		linkout += self.processTable(data, info.banktbl)
		linkout += self.processTable(data, info.grouptbl)
		write(linkout, self.bcsar.outdir + 'ExternalLinkedFiles.txt')
	
	def processTable(self, data, table):
		linked = []
		for entry in table.entries:
			fileentry = self.files[entry.fileid]
			if fileentry.valid:
				if fileentry.internal:
					data.seek(fileentry.offset + self.offset + 8)
					bwrite(data.read(fileentry.size), self.makepath(entry))
				else:
					linked.append((entry.name, fileentry.path))
			else:
				error.InvalidDataWarning('Invalid entry %d' % entry.fileid)
				continue
		linkout = ''
		for entry in linked:
			linkout += '%s: %s\n' % entry
		return linkout
			
	
	def makepath(self, entry):
		pathes = {
			'BCWAR': self.bcsar.warcdir,
			'BCWSD': self.bcsar.wsddir,
			'BCBNK': self.bcsar.bankdir,
			'BCGRP': self.bcsar.groupdir,
			'BCSEQ': self.bcsar.seqdir,
		}
		filename = path(pathes[entry.type], '%s.%s' % (entry.name, entry.type))
		return filename

class extractBCSAR (rawutil.TypeReader):
	def __init__(self, filename, file, verbose, opts={}):
		self.outdir = make_outdir(filename)
		self.make_outdirs()
		self.verbose = verbose
		self.data = file
		self.data.seek(0)
		INFOTable.bcsar = self
		BCSARSection.bcsar = self
		self.strg = STRG()
		self.info = INFO()
		self.file = FILE()
		INFOTable.info = self.info
		self.readheader()
		self.strg.extract(self.data)
		self.info.extract(self.data)
	
	def make_outdirs(self):
		self.warcdir = makedirs(self.outdir + 'WaveArchives')
		self.bankdir = makedirs(self.outdir + 'Banks')
		self.groupdir = makedirs(self.outdir + 'Groups')
		self.seqdir = makedirs(self.outdir + 'Sequences')
		self.wsddir = makedirs(self.outdir + 'WaveSounds')
	
	def readheader(self):
		self.byteorder = ENDIANS[self.unpack_from('>H', self.data, 4)[0]]
		INFOTable.byteorder = self.byteorder
		BCSARSection.byteorder = self.byteorder
		header = self.unpack_from(CSAR_HEADER, self.data, 0)
		magic = header[0]
		if magic != b'CSAR':
			error.InvalidMagicError('Invalid magic %s, expected CSAR' % byterepr(magic))
		#bom = header[1]
		#headerlen = header[2]
		#version = header[3]  #?
		#filelen = header[4]
		seccount = header[5]
		#padding = header[6]  #?
		for sec in header[7]:
			id, padding, offset, size = sec
			if id == STRG.id:  #STRG
				self.strg.offset = offset
				self.strg.size = size
			elif id == INFO.id:  #INFO
				self.info.offset = offset
				self.info.size = size
			elif id == FILE.id:  #FILE
				self.file.offset = offset
				self.file.size = size
			else:
				error.InvalidSectionError('Invalid section ID 0x%04x' % id)

	def extract(self):
		self.file.extract(self.data)
