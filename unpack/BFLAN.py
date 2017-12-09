# -*- coding:utf-8 -*-
import os
from util import error, ENDIANS
from util.funcops import byterepr, ClsFunc
from util.fileops import *
import util.rawutil as rawutil
from util.txtree import dump
from collections import OrderedDict

FLAN_HEADER = '4s2H 2HI2H'
pat1_SECTION = '4sI 2H2I2HBU'
pai1_SECTION = '4sI H2B2HI /5[I]'
pai1_ENTRY = 'n28a2BH /1[I]'
pai1_TAG = '4sBU /1[I]'
pai1_TAGENTRY = '2B3HI'

FLPA_types = ('X-trans', 'Y-trans', 'Z-trans', 'X-rotate', 'Y-rotate', 'Z-rotate', 'X-scale', 'Y-scale', 'X-size', 'Y-size')

FLVI_types = ('visible')

FLTP_types = ('texture-pattern')

FLVC_types = ('LT-red', 'LT-green', 'LT-blue', 'LT-alpha', 'RT-red', 'RT-green', 'RT-blue', 'RT-alpha', 'LB-red', 'LB-green', 'LB-blue', 'LB-alpha', 'RB-red', 'RB-green', 'RB-blue', 'RB-alpha', 'pane-alpha')

FLMC_types = ('BlackColor-red', 'BlackColo-green', 'BlackColor-blue', 'BlackColor-alpha', 'WhiteColor-red', 'WhiteColor-green', 'WhiteColor-blue', 'WhiteColor-alpha')

FLTS_types = ('U-trans', 'V-trans', 'rotate', 'U-scale', 'V-scale')

FLIM_types = ('rotate', 'U-scale', 'V-scale')


class extractBFLAN (rawutil.TypeReader, ClsFunc):
	def main(self, filename, file, verbose, opts={}):
		self.outfile = make_outfile(filename, 'tflan')
		self.verbose = verbose
		self.file = file
		self.tree = OrderedDict()
		self.tree['BFLAN'] = OrderedDict()
		self.root = self.tree['BFLAN']
		ptr = self.readheader()
		self.extract_sections(ptr)
		write(dump(self.tree), self.outfile)
	
	def readheader(self):
		self.byteorder = ENDIANS[rawutil.unpack_from('>H', self.file, 4)[0]]
		header = self.unpack_from(FLAN_HEADER, self.file, 0)
		if header[0] != b'FLAN':
			error.InvalidMagicError('Invalid magic %s, expected FLAN' % byterepr(header[0]))
		#bom = header[1]
		headerlen = header[2]
		self.version = header[3]
		#unknown = header[4]
		self.filelen = header[5]
		self.seccount = header[6]
		#padding = header[7]
		return headerlen
	
	def extract_sections(self, ptr):
		for i in range(self.seccount):
			magic, seclen = self.unpack_from('4sI', self.file, ptr)
			magic = magic.decode('ascii')
			method = eval('self.read%s' % magic)
			method(ptr)
			ptr += seclen
			
	def readpat1(self, ptr):
		data = self.unpack_from(pat1_SECTION, self.file, ptr)
		magic = data[0]
		if magic != b'pat1':
			error.InvalidMagicError('Invalid pat1 magic %s' % byterepr(magic))
		seclen = data[1]
		order = data[2]
		seconds = data[3]
		firstoffset = data[4]
		secondsoffset = data[5]
		start = data[6]
		end = data[7]
		childbinding = data[8]
		#padding = data[9]
		first = self.unpack_from('n', self.file, ptr + firstoffset)[0].decode('ascii')
		groups = []
		for i in range(seconds):
			groups.append(self.unpack_from('n', self.file, ptr + secondsoffset + i * 28)[0].decode('ascii'))
		self.root['pat1'] = OrderedDict()
		node = self.root['pat1']
		node['order'] = order
		node['seconds-number'] = seconds
		node['start'] = start
		node['end'] = end
		node['child-binding'] = childbinding
		node['first-group'] = first
		node['seconds-group-names'] = groups
	
	def readpai1(self, ptr):
		data = self.unpack_from(pai1_SECTION, self.file, ptr)
		magic = data[0]
		if magic != b'pai1':
			error.InvalidMagicError('Invalid pai1 magic %s' % byterepr(magic))
		seclen = data[1]
		framesize = data[2]
		flags = data[3]
		#padding = data[4]
		timgcount = data[5]
		entrycount = data[6]
		entryoffset = data[7]
		timgoffsets = data[8]
		self.root['pai1'] = OrderedDict()
		node = self.root['pai1']
		node['frame-size'] = framesize
		node['flags'] = flags
		timgs = []
		for (offset, ) in timgoffsets:
			timgs.append(self.unpack_from('n', self.file, ptr + offset + 20)[0].decode('ascii'))
		node['timgs'] = timgs
		offsets = []
		for i in range(entrycount):
			offsets.append(self.unpack_from('I', self.file, ptr + entryoffset + i * 4)[0])
		node['entries'] = []
		for offset in offsets:
			entry = OrderedDict()
			data = self.unpack_from(pai1_ENTRY, self.file, offset + ptr)
			entry['name'] = data[0]
			tagcount = data[1]
			ismaterial = data[2]
			#padding = data[3]
			tagoffsets = data[4]
			entry['tags'] = []
			for (tagoffset, ) in tagoffsets:
				tagpos = ptr + offset + tagoffset
				tag = OrderedDict()
				data = self.unpack_from(pai1_TAG, self.file, tagpos)
				type = data[0].decode('ascii')
				entrycount = data[1]
				#padding = data[2]
				suboffsets = data[3]
				tag['type'] = type
				tag['entries'] = []
				for (suboffset, ) in suboffsets:
					pos = tagpos + suboffset
					tagentry = OrderedDict()
					data = self.unpack_from(pai1_TAGENTRY, self.file, pos)
					type1 = data[0]
					type2 = data[1]
					datatype = data[2]
					coordcount = data[3]
					totag = data[4]
					tagentry['type1'] = type1
					tagentry['type2'] = self.gettype(type2, type)
					tagentry['data-type'] = datatype
					tagentry['coord-count'] = coordcount
					tagentry['offset-to-tag'] = rawutil.hex(totag, 8)
					if datatype == 0x100:
						tagentry['frame'], tagentry['data2'] = self.unpack_from('2f', self.file)
					elif datatype == 0x200:
						tagentry['frame'], tagentry['value'], tagentry['blend'] = self.unpack_from('3f', self.file)
					tag['entries'].append(tagentry)
				entry['tags'].append(tag)
			node['entries'].append(entry)
	
	def gettype(self, type, tagtype):
		el = eval('%s_types[type]' % tagtype)
		return el
