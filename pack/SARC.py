# -*- coding:utf-8 -*-
from util.rawutil import TypeWriter
from util.fileops import bread, bwrite, basedir
from util.funcops import ClsFunc
from util import error, BOMS


SARC_HEADER_STRUCT = '4s2H3I'
SFAT_HEADER_STRUCT = '4s2HI'
SFAT_NODE_STRUCT = '4I'
SFNT_HEADER_STRUCT = '4s2H'


class SFATnode (object):
	def __init__(self):
		self.hash = 0
		self.has_name = 1
		self.name_offset = 0
		self.data_start = 0
		self.data_end = 0


class packSARC (ClsFunc, TypeWriter):
	def main(self, filenames, outname, endian, verbose, opts={}, embedded=False):
		self.byteorder = endian
		self.verbose = verbose
		self.embedded = embedded  #Used for embedded SARC sections in other files such as ALYT
		self.hash_multiplier = 0x65
		self.outname = outname
		self.make_nodes(filenames)
		return self.repack_sections()
	
	def make_nodes(self, filenames):
		self.nodes = []
		for filename in filenames:
			node = SFATnode()
			if filename.endswith('.noname.bin'):
				node.has_name = 0
				node.name = b''
				node.hash = int(filename.lstrip('0x').replace('.noname.bin', ''), 16)
			else:
				node.has_name = 1
				node.name = filename
				node.hash = self.calc_hash(filename)
			self.nodes.append(node)
		self.nodes.sort(key=lambda node: node.hash)
	
	def repack_sections(self):
		sfat = b''
		sfnt = b''
		data = b''
		for node in self.nodes:
			name_offset = len(sfnt) // 4
			data_start = len(data)
			data += bread(node.name)
			data_end = len(data)
			data += self.align(data, 0x80)
			sfnt += self.pack('n', node.name)
			sfnt += self.align(sfnt, 4)
			sfat += self.pack(SFAT_NODE_STRUCT, node.hash, name_offset, data_start, data_end)
		sfat = self.pack(SFAT_HEADER_STRUCT, b'SFAT', 12, len(self.nodes), self.hash_multiplier) + sfat
		sfnt += self.align(sfat, 0x100)
		sfnt = self.pack(SFNT_HEADER_STRUCT, b'SFNT', 8, 0) + sfnt
		sfnt += self.align(sfnt, 0x100)
		meta = sfat + sfnt
		final = meta + data
		sarchdr = self.pack(SARC_HEADER_STRUCT, b'SARC', 20, 0xfeff, len(final) + 20, len(meta) + 20, 0x00000100)
		final = sarchdr + final
		if self.embedded:
			return final
		else:
			basedir()
			bwrite(final, self.outname)
	
	def calc_hash(self, name):
		result = 0
		for c in name:
			result = ord(c) + (result * self.hash_multiplier)
			result &= 0xFFFFFFFF
		return result
