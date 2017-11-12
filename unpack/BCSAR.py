# -*- coding:utf-8 -*-
import os
from util import error, ENDIANS
from util.funcops import getsup, byterepr
from util.fileops import *
import util.rawutil as rawutil

CSAR_HEADER_STRUCT = '4s2H13I'
STRG_HEADER_STRUCT = '4s5I'
INFO_HEADER_STRUCT = '4s17I'
INFO_SECTION_STRUCT = 'I/0[2I]$'

BANK_ENTRY_STRUCT = '6I'
PLAYER_ENTRY_STRUCT = '4I'
WAR_ENTRY_STRUCT = '3I'
GROUP_ENTRY_STRUCT = '3I'


class FileNode (object):
	def __init__(self):
		self.name = ''
		self.filename = None
		self.internal = False
		self.type = 'Undefined'
		self.fileid = None
		self.bank = None
		self.set = None
		self.war = None
		self.group = None


class BankNode (object):
	def __init__(self):
		self.fileid = None
		self.name = ''


class AudioNode (object):
	def __init__(self):
		self.fileid = None
		self.name = ''


class SetNode (object):
	def __init__(self):
		self.fileid = None
		self.name = ''


class WARNode (object):
	def __init__(self):
		self.fileid = None
		self.name = ''


class GroupNode (object):
	def __init__(self):
		self.fileid = None
		self.name = ''


class extractBCSAR (rawutil.TypeReader):
	def __init__(self, filename, data, verbose, opts={}):
		self.make_outdirs(filename)
		self.verbose = verbose
		self.byteorder = ENDIANS[rawutil.unpack_from('>H', data, 4)[0]]
		self.readheader(data)
		self.readSTRG()
		self.readINFO()

	def make_outdirs(self, filename):
		self.outdir = make_outdir(filename)
		self.wardir = makedirs(self.outdir + 'WaveArchives')
		self.bankdir = makedirs(self.outdir + 'Banks')
		self.groupdir = makedirs(self.outdir + 'Groups')
		self.seqdir = makedirs(self.outdir + 'Sequences')
		self.wsddir = makedirs(self.outdir + 'WaveSounds')

	def readheader(self, data):
		hdata = self.unpack_from(CSAR_HEADER_STRUCT, data, 0)
		if hdata[0] != b'CSAR':
			error('Invalid magic %s, expected CSAR' % byterepr(hdata[0]), 301)
		#bom = hdata[1]
		#headerlen = hdata[2]
		self.version = hdata[3]
		#filelen = hdata[4]
		#seccount = hdata[5]
		i = 6
		while i < 14:
			id, offset, length = hdata[i:i + 3]
			i += 3
			secdata = data[offset:offset + length]
			if id == 0x2000:
				self.strg = secdata
			elif id == 0x2001:
				self.info = secdata
			elif id == 0x2002:
				self.file = secdata
				self.fileoffset = offset
			else:
				error('Unsupported partition 0x%04x' % id, 303)

		#bwrite(self.strg, 'STRG')
		#bwrite(self.info, 'INFO')
		#bwrite(self.file, 'FILE')

	def readSTRG(self):
		hdata, ptr = self.unpack_from(STRG_HEADER_STRUCT, self.strg, 0, getptr=True)
		if hdata[0] != b'STRG':
			error('Invalid STRG magic: %s' % byterepr(hdata[0]), 301)
		#strglen = hdata[1]
		i = 2
		while i < 6:
			id, offset = hdata[i:i + 2]
			i += 2
			if id == 0x2400:
				fntoffset = offset + 8
			elif id == 0x2401:
				tbloffset = offset + 8
		strcount = self.unpack_from('I', self.strg, fntoffset)[0]
		strings = []
		for i in range(0, strcount):
			id, nameoffset, namelen = self.unpack_from('3I', self.strg, fntoffset + (i * 12) + 4)
			name = self.strg[fntoffset + nameoffset:fntoffset + nameoffset + namelen].rstrip(b'\x00').decode('utf-8')
			strings.append(name)
		self.strings = strings

	def readINFO(self):
		hdata, ptr = self.unpack_from(INFO_HEADER_STRUCT, self.info, 0, getptr=True)
		if hdata[0] != b'INFO':
			error('Invalid INFO magic: %s' % byterepr(hdata[0]), 301)
		infolen = hdata[1]
		infooffsets = [None] * 7
		i = 2
		while i < 16:
			id, offset = hdata[i:i + 2]
			i += 2
			if id == 0x2100:
				infooffsets[0] = offset + 8
			elif id == 0x2101:
				infooffsets[1] = offset + 8
			elif id == 0x2102:
				infooffsets[2] = offset + 8
			elif id == 0x2103:
				infooffsets[3] = offset + 8
			elif id == 0x2104:
				infooffsets[4] = offset + 8
			elif id == 0x2105:
				infooffsets[5] = offset + 8
			elif id == 0x2106:
				infooffsets[6] = offset + 8
			else:
				error('Unsupported INFO section %x' % id, 303)
		#unknown = hdata[16]
		#unknown = hdata[18]
		infosecs = [None] * 7
		for i in range(0, len(infooffsets)):
			startoffset = infooffsets[i]
			try:
				endoffset = getsup(infooffsets, startoffset)
			except IndexError:
				endoffset = infolen
			infosecs[i] = self.info[startoffset: endoffset]
		self.readFileTable(infosecs[6])
		self.readBankTable(infosecs[1])
		self.readAudioTable(infosecs[0])
		self.readPlayerTable(infosecs[2])
		self.readWARTable(infosecs[3])
		self.readSetTable(infosecs[4])
		self.readGroupTable(infosecs[5])

	def readINFOsec(self, data):
		entrycount, otbl, entrydata = self.unpack(INFO_SECTION_STRUCT, data)
		tbllen = len(data) - len(entrydata) - 4
		offsets = [el[1] - tbllen - 4 for el in otbl]
		return entrycount, offsets, entrydata

	def readAudioTable(self, data):
		entrycount, offsets, entrydata = self.readINFOsec(data)
		self.audios = [AudioNode() for i in range(0, entrycount)]
		for i, offset in enumerate(offsets):
			fileidx, ptr = self.uint32(entrydata, offset)
			type, ptr = self.uint32(entrydata, ptr + 8)
			stridx, ptr = self.uint32(entrydata, ptr + 8)
			self.audios[i].fileid = fileidx
			self.files[fileidx].name = self.strings[stridx]
			self.audios[i].name = self.strings[stridx]
			self.files[fileidx].fileid = i
			if type == 0x2201:  #external stream
				self.files[fileidx].type = self.audios[i].type = 'bcstm'
			elif type == 0x2202:  #CWSD
				self.files[fileidx].type = self.audios[i].type = 'bcwsd'
			elif type == 0x2203:
				self.files[fileidx].type = self.audios[i].type = 'bcseq'
				self.audios[i].bank, ptr = self.uint16(entrydata, ptr + 0x44)

	def readBankTable(self, data):
		entrycount, offsets, entrydata = self.readINFOsec(data)
		banktable = [self.unpack_from(BANK_ENTRY_STRUCT, entrydata, offset) for offset in offsets]
		self.banks = [BankNode() for i in range(0, entrycount)]
		for i, entry in enumerate(banktable):
			fileidx = entry[0]
			self.files[fileidx].type = 'bcbnk'
			self.files[fileidx].bank = i
			self.banks[i].fileid = fileidx
			stridx = entry[4]
			self.banks[i].name = self.strings[stridx]
			self.files[fileidx].name = self.strings[stridx]

	def readPlayerTable(self, data):
		entrycount, offsets, entrydata = self.readINFOsec(data)
		self.playertable = [self.unpack_from(PLAYER_ENTRY_STRUCT, entrydata, offset) for offset in offsets]

	def readWARTable(self, data):
		entrycount, offsets, entrydata = self.readINFOsec(data)
		self.wars = [WARNode() for i in range(0, entrycount)]
		for i, offset in enumerate(offsets):
			fileidx, ptr = self.uint32(entrydata, offset)
			self.wars[i].fileid = fileidx
			self.files[fileidx].war = i
			self.files[fileidx].type = 'bcwar'
			entrylen = (offsets[i + 1] - offset) if i != len(offsets) - 1 else (len(entrydata) - offset)
			if entrylen > 0x0c:
				self.wars[i].name = self.uint32(entrydata, ptr)
			else:
				self.wars[i].name = 'WAR_%08x' % i
			self.files[fileidx].name = self.wars[i].name

	def readSetTable(self, data):
		entrycount, offsets, entrydata = self.readINFOsec(data)
		entries = []
		self.sets = [SetNode() for i in range(0, entrycount)]
		for i, offset in enumerate(offsets):
			hdr, ptr = self.uint32(entrydata, offset)
			if hdr == 0xffffffff:
				continue
			stridx, ptr = self.uint32(entrydata, ptr + 0x18)
			self.sets[i].name = self.strings[stridx]
			fileidx, ptr = self.uint32(entrydata, ptr + 4)
			self.sets[i].fileid = fileidx
			self.files[fileidx].set = i

	def readGroupTable(self, data):
		entrycount, offsets, entrydata = self.readINFOsec(data)
		grouptable = [self.unpack_from(GROUP_ENTRY_STRUCT, entrydata, offset) for offset in offsets]
		self.groups = [GroupNode() for i in range(0, entrycount)]
		for i, entry in enumerate(grouptable):
			fileidx, unknown, stridx = entry
			self.groups[i].fileid = fileidx
			self.groups[i].name = self.strings[stridx]
			self.files[fileidx].group = i
			self.files[fileidx].name = self.strings[stridx]
			self.files[fileidx].type = 'bcgrp'

	def readFileTable(self, data):
		entrycount, offsets, entrydata = self.readINFOsec(data)
		self.files = [FileNode() for i in range(0, entrycount)]
		for i, offset in enumerate(offsets):
			id, ptr = self.uint32(entrydata, offset)
			if id == 0x220d:  #external file
				nameloc, ptr = self.uint32(entrydata, ptr)
				self.files[i].filename, ptr = self.string(entrydata, offset + nameloc)
			elif id == 0x220c:
				self.files[i].internal = True
				dataloc, ptr = self.uint32(entrydata, ptr)
				if dataloc == 0xffffffff:
					continue
				type, ptr = self.uint32(entrydata, dataloc + offset)
				self.files[i].offset, ptr = self.uint32(entrydata, ptr)
				self.files[i].offset += 8
				self.files[i].size, ptr = self.uint32(entrydata, ptr)

	def extract(self):
		linked = []
		for i, file in enumerate(self.files):
			if file.internal:
				if file.name == '':
					print('file %x has no name' % i)
					continue
				elif file.type == 'Undefined':
					print('file %x type is not defined' % i)
				filename = '%s.%s' % (file.name, file.type)
				filedata = self.file[file.offset: file.offset + file.size]
				folder = self.get_folder(file.type)
				bwrite(filedata, folder + filename)
			else:
				if file.filename is not None:
					linked.append(file.filename)
				else:
					print('external file %x has no file name' % i)
					continue
		write('\n'.join(linked), self.outdir + 'LinkedExternalStreams.txt')
		f, b, w, a = self.files, self.banks, self.wars, self.audios

	def get_folder(self, type):
		if type == 'bcseq':
			return self.seqdir
		elif type == 'bcwsd':
			return self.wsddir
		elif type == 'bcwar':
			return self.wardir
		elif type == 'bcbnk':
			return self.bankdir
		elif type == 'bcgrp':
			return self.groupdir
		else:
			return self.outdir
