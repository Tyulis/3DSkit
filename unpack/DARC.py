# -*- coding:utf-8 -*-
import os.path
from util import error, ENDIANS
from util.fileops import make_outdir, bread, bwrite, makedirs
import util.rawutil as rawutil

DARC_HEADER_STRUCT = '4s4H4I'
DARC_TABLE_STRUCT = '#0[3I]'


class DARCTableEntry (object):
	def __init__(self, entry):
		self.nameoffset = entry[0] & 0xffffff
		self.isdir = entry[0] >> 24
		self.dataoffset = entry[1]
		self.datalength = entry[2]
		self.name = ''


class extractDARC (rawutil.TypeReader):
	def __init__(self, filename, data):
		self.outdir = make_outdir(filename)
		ptr = self.readhdr(data)
		ptr = self.readtable(data, ptr)
		self.data = data
	
	def readhdr(self, data):
		magic, endian = rawutil.unpack_from('>4sH', data, 0)
		if magic != b'darc':
			error('Invalid magic : %s' % magic)
		self.byteorder = ENDIANS[endian]
		hdr, ptr = self.unpack_from(DARC_HEADER_STRUCT, data, 0, getptr=True)
		#magic=hdr[0]
		#bom=hdr[1]
		#headerlen=hdr[2]
		#unknown = hdr[3]
		self.version = hdr[4]
		#filelen=hdr[5]
		self.tableoffset = hdr[6]
		self.tablelen = hdr[7]
		self.dataoffset = hdr[8]
		return ptr
	
	def readtable(self, data, ptr):
		rawtable = data[self.tableoffset:self.tableoffset + self.tablelen]
		root = self.unpack_from('3I', data, ptr)
		entrynum = root[2]
		tablebs, ptr = self.unpack_from(DARC_TABLE_STRUCT, data, ptr, refdata=[entrynum], getptr=True)
		self.table = [DARCTableEntry(entry) for entry in tablebs[0]]
		for i, entry in enumerate(self.table):
			self.table[i].name, end = self.utf16string(data, entry.nameoffset + ptr)
	
	def list(self):
		print('Directories:')
		print('\n'.join([entry.name for entry in self.table if entry.isdir]))
		print('')
		print('Files:')
		print('\n'.join([entry.name for entry in self.table if not entry.isdir]))
	
	def extract(self):
		actfolder = [self.outdir]
		folderend = -1
		for i, entry in enumerate(self.table):
			if i == folderend:
				actfolder.pop()
			if entry.isdir and entry.name not in ('', '.'):
				actfolder.append(entry.name)
				makedirs(os.path.join(*(actfolder + ['a'])))  #ugly
				folderend = entry.datalength
			elif not entry.isdir:  #because of null and .
				filedata = self.data[entry.dataoffset:entry.dataoffset + entry.datalength]
				path = os.path.join(*(actfolder + [entry.name]))
				bwrite(filedata, path)
