# -*- coding:utf-8 -*-
import os
import compress
from util import error, ENDIANS
from util.fileops import make_outdir, bread, bwrite, makedirs
import util.rawutil as rawutil
from ._formats import get_ext

GARC_HEADER_STRUCT = '4sI2H3I'
GARC_HEADER_END = {
	0x0400: 'I',
	0x0600: '3I'
}
GARC_FATO_SECTION = '4sI2H/2[I]'
GARC_FATB_HEADER = '4s2I'


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


class extractGARC (rawutil.TypeReader):
	def __init__(self, filename, data):
		self.outdir = make_outdir(filename)
		ptr = self.readheader(data)
		ptr = self.readFATO(data, ptr)
		ptr = self.readFATB(data, ptr)
		self.data = data
		self.ptr = ptr
	
	def readheader(self, data):
		self.byteorder = ENDIANS[rawutil.unpack_from('>H', data, 8)[0]]
		hdata, ptr = self.unpack_from(GARC_HEADER_STRUCT, data, 0, getptr=True)
		if hdata[0] != b'CRAG':
			error('Invalid magic : %s' % hdata[0])
		#headerlen = hdata[1]
		#bom = hdata[2]
		self.version = hdata[3]
		self.chunkcount = hdata[4]
		self.dataoffset = hdata[5]
		#filelen = hdata[6]
		padinfo, ptr = self.unpack_from(GARC_HEADER_END[self.version], data, ptr, getptr=True)
		if self.version == 0x0400:
			print('Version: 4')
			self.larger_unpadded = padinfo[0]
			self.pad_to_nearest = 4
		elif self.version == 0x0600:
			print('Version: 6')
			self.larger_padded = padinfo[0]
			self.larger_unpadded = padinfo[1]
			self.pad_to_nearest = padinfo[2]
		return ptr
	
	def readFATO(self, data, ptr):
		fato, ptr = self.unpack_from(GARC_FATO_SECTION, data, ptr, getptr=True)
		if fato[0] != b'OTAF':
			error('Invalid FATO magic : %s' % fato[0])
		#headerlen = fato[1]
		entrycount = fato[2]
		self.filenum = entrycount
		#padding = fato[3]
		table = fato[4]
		self.fato = [el[0] for el in table]
		return ptr
	
	def readFATB(self, data, ptr):
		hdr, ptr = self.unpack_from(GARC_FATB_HEADER, data, ptr, getptr=True)
		if hdr[0] != b'BTAF':
			error('Invalid FATB magic : %s' % hdr[0])
		#headerlen = hdr[1]
		entrycount = hdr[2]
		self.fatb = []
		for i in range(0, entrycount):
			entry = FATBEntry()
			flags, ptr = self.uint32(data, ptr)
			entry.flags = flags
			for j in range(0, 32):
				exists = (flags & 1) == 1
				flags >>= 1
				if exists:
					subentry = FATBSubEntry()
					subentry.start, ptr = self.uint32(data, ptr)
					subentry.end, ptr = self.uint32(data, ptr)
					subentry.length, ptr = self.uint32(data, ptr)
					entry.subentries.append(subentry)
			if len(entry.subentries) > 1:
				entry.isfolder = True
			self.fatb.append(entry)
		return ptr
					
	def list(self):
		print('%d files found' % self.filenum)
	
	def extract(self):
		data = self.data
		ptr = self.ptr
		for i, entry in enumerate(self.fatb):
			if entry.isfolder:
				outpath = self.outdir + str(i) + os.path.sep
				try: os.mkdir(outpath)
				except: pass
				os.makedirs(outpath)
				for j, subentry in enumerate(entry.subentries):
					start = subentry.start + self.dataoffset
					end = start + subentry.length
					filedata = data[start:end]
					comp = compress.recognize(filedata)
					ext = get_ext(filedata)
					outname = outpath + str(j) + ext
					if comp == 'LZ11':
						filedata = compress.decompress(filedata, comp, self.byteorder)
						ext = get_ext(filedata)
						outname = outpath + 'dec_' + str(j) + ext
					bwrite(filedata, outname)
			else:
				subentry = entry.subentries[0]
				start = subentry.start + self.dataoffset
				end = start + subentry.length
				filedata = data[start:end]
				comp = compress.recognize(filedata)
				ext = get_ext(filedata)
				outname = self.outdir + str(i) + ext
				if comp == 'LZ11':
					filedata = compress.decompress(filedata, comp)
					ext = get_ext(filedata)
					outname = self.outdir + 'dec_' + str(i) + ext
				bwrite(filedata, outname)
