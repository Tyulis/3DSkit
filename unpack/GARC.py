# -*- coding:utf-8 -*-
import os
import compress
from util import error, ENDIANS
from util.fileops import *
from util.funcops import byterepr
import util.rawutil as rawutil
from ._formats import get_ext
from io import BytesIO

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
	def __init__(self, filename, data, verbose, opts={}):
		self.outdir = make_outdir(filename)
		self.verbose = verbose
		ptr = self.readheader(data)
		ptr = self.readFATO(data, ptr)
		ptr = self.readFATB(data, ptr)
		self.data = data
		self.ptr = ptr
	
	def readheader(self, data):
		self.byteorder = ENDIANS[rawutil.unpack_from('>H', data, 8)[0]]
		hdata, ptr = self.unpack_from(GARC_HEADER_STRUCT, data, 0, getptr=True)
		if hdata[0] != b'CRAG':
			error('Invalid magic %s, expected CRAG' % byterepr(hdata[0]), 301)
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
		else:
			error('Unsupported version 0x%04x' % self.version, 304)
		return ptr
	
	def readFATO(self, data, ptr):
		fato, ptr = self.unpack_from(GARC_FATO_SECTION, data, ptr, getptr=True)
		if fato[0] != b'OTAF':
			error('Invalid FATO magic %s, expected OTAF' % byterepr(fato[0]), 301)
		#sectionlen = fato[1]
		entrycount = fato[2]
		self.filenum = entrycount
		#padding = fato[3]
		table = fato[4]
		self.fato = [el[0] for el in table]
		if self.verbose:
			print('File count: %d' % entrycount)
		return ptr
	
	def readFATB(self, data, ptr):
		hdr, ptr = self.unpack_from(GARC_FATB_HEADER, data, ptr, getptr=True)
		if hdr[0] != b'BTAF':
			error('Invalid FATB magic %s, expected BTAF' % byterepr(hdr[0]), 301)
		#headerlen = hdr[1]
		entrycount = hdr[2]
		tblstart = ptr
		self.fatb = []
		#for i in range(0, entrycount):
		for offset in self.fato:
			entry = FATBEntry()
			flags, ptr = self.unpack_from('I', data, tblstart + offset, getptr=True)
			flags = flags[0]
			entry.flags = flags
			for j in range(0, 32):
				exists = (flags & 1)
				flags >>= 1
				if exists:
					subentry = FATBSubEntry()
					subdata, ptr = self.unpack_from('3I', data, ptr, getptr=True)
					subentry.start, subentry.end, subentry.length = subdata
					entry.subentries += [None] * (j - len(entry.subentries) + 1)
					entry.subentries[j] = subentry
			entry.isfolder = len(entry.subentries) > 1
			self.fatb.append(entry)
		return ptr
	
	def extract(self):
		data = self.data
		for i, entry in enumerate(self.fatb):
			if entry.isfolder:
				outpath = self.outdir + str(i) + os.path.sep
				try:
					os.mkdir(outpath)
				except:
					pass
				for j, subentry in enumerate(entry.subentries):
					if subentry is None:
						continue
					start = subentry.start + self.dataoffset
					data.seek(start)
					filedata = BytesIO(data.read(subentry.length))
					comp = compress.recognize(filedata)
					filedata.seek(0)
					ext = get_ext(filedata)
					outname = outpath + str(j) + ext
					if comp == 'LZ11':
						outname = outpath + 'dec_' + str(j) + ext
						out = open(outname, 'wb+')
						result = compress.decompress(filedata, out, comp, self.verbose)
						if result == 901:  #Ugly
							outname = outpath + str(j) + ext
							print('For %s' % outname)
							filedata.seek(0)
							bwrite(filedata.read(), outname)
					else:
						bwrite(filedata.read(), outname)
			else:
				subentry = entry.subentries[0]
				start = subentry.start + self.dataoffset
				data.seek(start)
				filedata = BytesIO(data.read(subentry.length))
				comp = compress.recognize(filedata)
				filedata.seek(0)
				ext = get_ext(filedata)
				outname = self.outdir + str(i) + ext
				if comp == 'LZ11':
					outname = self.outdir + 'dec_%d%s' % (i, ext)
					out = open(outname, 'wb+')
					result = compress.decompress(filedata, out, comp, self.verbose)
					out.close()
					if result == 901:  #Ugly
						os.remove(outname)
						outname = self.outdir + str(i) + ext
						print('For %s' % outname)
						filedata.seek(0)
						bwrite(filedata.read(), outname)
				else:
					bwrite(filedata.read(), outname)
