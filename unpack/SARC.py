# -*- coding:utf-8 -*-
import os.path
from util import error, ENDIANS
from util.fileops import make_outdir, bread, bwrite, makedirs
import util.rawutil as rawutil

SARC_HEADER_STRUCT = '4s2H3I'
SFAT_STRUCT = '4s2HI/2[4I]'
SFNT_HEADER_STRUCT = '4s2H'


class SFATnode (object):
	def __init__(self, node):
		self.hash = node[0]
		self.has_name = node[1] >> 24
		self.name_offset = (node[1] & 0xffffff) * 4
		self.data_start = node[2]
		self.data_end = node[3]


class extractSARC (rawutil.TypeReader):
	def __init__(self, filename, data):
		self.outdir = make_outdir(filename)
		ptr = self.readheader(data)
		ptr = self.readSFAT(data, ptr)
		self.readSFNT(data, ptr)
		self.data = data
	
	def readheader(self, data):
		magic, hdrlen, bom = rawutil.unpack_from('>4s2H', data, 0)
		if magic != b'SARC':
			error('Not a valid SARC file.')
		self.byteorder = ENDIANS[bom]
		hdr, ptr = self.unpack_from(SARC_HEADER_STRUCT, data, 0, getptr=True)
		#magic = hdr[0]
		#headerlen = hdr[1]
		#bom = hdr[2]
		#filelen = hdr[3]
		self.dataoffset = hdr[4]
		#unknown = hdr[5]
		return ptr
	
	def readSFAT(self, data, ptr):
		sfat, ptr = self.unpack_from(SFAT_STRUCT, data, ptr, getptr=True)
		magic = sfat[0]
		if magic != b'SFAT':
			error('Issue with SFAT: invalid magic %s' % magic.decode('ascii'))
		#headerlen = sfat[1]
		self.node_count = sfat[2]
		self.hash_multiplier = sfat[3]
		self.nodes = [SFATnode(node) for node in sfat[4]]
		return ptr
	
	def readSFNT(self, data, ptr):
		hdr, ptr = self.unpack_from(SFNT_HEADER_STRUCT, data, ptr, getptr=True)
		magic = hdr[0].decode('ascii')
		if magic != 'SFNT':
			error('Issue with SFNT: invalid magic %s' % magic)
		#headerlen = hdr[1]
		#unknown = hdr[2]
		for i, node in enumerate(self.nodes):
			if node.has_name:
				filename = self.string(data, node.name_offset + ptr)[0]
				node.name = filename
				if self.calc_hash(filename) != node.hash:
					error('File name %s doesn\'t correspond to his hash %08x' % (filename, node.hash))
			else:
				node.name = '0x%08x.noname.bin' % node.hash
			self.nodes[i] = node
	
	def extract(self):
		for node in self.nodes:
			filedata = self.data[self.dataoffset + node.data_start:self.dataoffset + node.data_end]
			makedirs(self.outdir + node.name)
			bwrite(filedata, self.outdir + node.name)
	
	def list(self):
		print('')
		for node in self.nodes:
			print(node.name + (' [fake name]' if not node.has_name else ''))
	
	def calc_hash(self, name):
		result = 0
		for c in name:
			result = ord(c) + (result * self.hash_multiplier)
			result &= 0xFFFFFFFF
		return result
