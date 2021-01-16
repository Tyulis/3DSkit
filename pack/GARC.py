# -*- coding:utf-8 -*-
from util import error
from util.utils import ClsFunc
from util.rawutil import TypeWriter
from util.filesystem import *
from compress.LZ11 import compressLZ11
from io import BytesIO
import __main__ as _main

GARC_HEADER_STRUCT = '4sI2H3I'
GARC_FATO_HEADER = '4sI2H'
GARC_FATB_HEADER = '4s2I'
HEADER_LENGTHS = {
	0x0400: 28,
	0x0600: 36,
}


class FATBEntry (object):
	def __init__(self):
		self.flags = 0
		self.subentries = []
		self.isfolder = False


class FATBSubEntry (object):
	def __init__(self):
		self.start = 0
		self.end = 0
		self.length = 0
		data = b""


class packGARC(ClsFunc, TypeWriter):
	def main(self, filenames, outname, endian, verbose, opts={}):
		self.byteorder = endian
		self.verbose = verbose
		if 'version' in opts:
			self.version = int(opts['version']) * 0x100
			if self.version not in (0x0400, 0x0600):
				error.InvalidOptionValue('You gave an unknown version number')
		else:
			self.version = 0x0600
		if outname.startswith(os.path.sep):
			self.file = open(outname, 'wb')
		else:
			self.file = open(_main.basedir + outname, 'wb')
		self.make_entries(filenames)
		self.pack_all(filenames)
		self.file.close()
	
	def make_entries(self, filenames):
		self.fimb = b''
		self.entries = {}
		larger = 0
		for name in filenames:
			file = open(name, 'rb')
			if 'dec_' in name:
				out = BytesIO()
				compressLZ11(file, out, self.verbose)
				out.seek(0)
				filedata = out.read()
			else:
				filedata = file.read()
			file.close()
			filelen = len(filedata)
			if filelen > larger:
				larger = filelen
			filedata += (4 - (len(filedata) % 4 or 4)) * b'\00'
			subentry = FATBSubEntry()
			subentry.length = filelen
			subentry.data = filedata
			if os.path.sep in name:
				foldername, _, filename = name.rpartition(os.path.sep)
				folder = int(foldername.rpartition(os.path.sep)[2])
				file = int(filename.split('.')[0].strip('dec_'))
				if folder not in self.entries.keys():
					entry = FATBEntry()
					entry.flags = 0
					entry.subentries = []
					entry.isfolder = True
					self.entries[folder] = entry
				self.entries[folder].subentries += [None] * (file - len(self.entries[folder].subentries) + 1)
				self.entries[folder].subentries[file] = subentry
				self.entries[folder].flags |= 1 << file
			else:
				entry = FATBEntry()
				entry.flags = 1
				entry.isfolder = False
				entry.subentries = [subentry]
				self.entries[int(name.split('.')[0].strip('dec_'))] = entry
		self.larger_unpadded = larger
		self.larger_padded = larger + (4 - (larger % 4 or 4))
		
	def pack_all(self, names):
		headerlen = HEADER_LENGTHS[self.version]
		self.pack(GARC_HEADER_STRUCT, b'CRAG', headerlen, 0xfeff, self.version, 4, 0, 0, self.file)  # '4sI2H3I'
		if self.version == 0x0600:
			self.pack('3I', self.larger_padded, self.larger_unpadded, 4, self.file)
		elif self.version == 0x0400:
			self.pack('I', self.larger_unpadded)
		
		self.entrycount = len(self.entries)
		self.subentrycount = sum([len([subentry for subentry in entry.subentries if subentry is not None]) for num, entry in self.entries.items()])
		
		self.fatooffset = headerlen
		self.file.seek(self.fatooffset)
		self.pack(GARC_FATO_HEADER, b'OTAF', 12, self.entrycount, 0xffff, self.file)
		self.fatooffset += 12
		self.fatboffset = (self.entrycount * 4) + headerlen + 12
		self.file.seek(self.fatboffset)
		self.pack(GARC_FATB_HEADER, b'BTAF', 12, self.subentrycount, self.file)
		self.fatboffset += 12
		self.dataoffset = self.packFAT() + 12
		self.file.seek(self.dataoffset - 12)
		self.pack('4s2I', b'BMIF', 12, len(self.fimb), self.file)
		self.file.write(self.fimb)
		self.file.seek(0, 2)
		filelen = self.file.tell()
		self.file.seek(16)
		self.pack('2I', self.dataoffset, filelen, self.file)
	
	def packFAT(self):
		self.fimb = b''
		actfatb = self.fatboffset
		actfato = self.fatooffset
		keys = sorted(list(self.entries.keys()))  #ugly.
		for num in keys:
			entry = self.entries[num]
			self.file.seek(actfato)
			self.pack('I', actfatb - self.fatboffset, self.file)
			actfato += 4
			self.file.seek(actfatb)
			self.pack('I', entry.flags, self.file)
			actfatb += 4
			for i, subentry in enumerate(entry.subentries):
				if subentry is None:
					continue
				start = len(self.fimb)
				self.fimb += subentry.data
				end = len(self.fimb)
				self.file.seek(actfatb)
				self.pack('3I', start, end, subentry.length, self.file)
				actfatb += 12
		self.file.seek(self.fatooffset - 8)
		self.pack('I', actfato - self.fatooffset + 12, self.file)
		self.file.seek(self.fatboffset - 8)
		self.pack('I', actfatb - self.fatboffset + 12, self.file)
		return actfatb
