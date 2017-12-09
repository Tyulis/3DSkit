# -*- coding:utf-8 -*-
from io import BytesIO
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
		if 'padding' in opts.keys():
			self.padding = int(opts['padding'])
		else:
			self.padding = 0x80
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
				node.name = filename.encode('utf-8')
				node.hash = self.calc_hash(filename)
			self.nodes.append(node)
		self.nodes.sort(key=lambda node: node.hash)
	
	def repack_sections(self):
		if self.embedded:
			file = BytesIO()
		else:
			file = open(self.outname, 'wb')
		file.write(bytes(20))  #Future header
		if self.verbose:
			print('Packing SFAT')
		self.pack(SFAT_HEADER_STRUCT, b'SFAT', 12, len(self.nodes), self.hash_multiplier, file)
		for node in self.nodes:
			self.pack('I', node.hash, file)
			node.sfatnameoffset = file.tell()
			file.write(bytes(12))
		#file.write(bytes(0x10 - (file.tell() % 0x10 or 0x10)))  #Always multiple of 0x10
		if self.verbose:
			print('Packing SFNT')
		self.pack(SFNT_HEADER_STRUCT, b'SFNT', 8, 0, file)
		sfntpos = file.tell()
		for node in self.nodes:
			node.nameoffset = (file.tell() - sfntpos) // 4
			file.write(node.name)
			file.write(bytes(4 - (file.tell() % 4)))  #Not "or 4", adds 4B at the end if already aligned 
		file.write(bytes(0x80 - (file.tell() % 0x80 or 0x80)))
		if self.verbose:
			print('Packing data')
		datastart = file.tell()
		for node in self.nodes:
			filedata = bread(node.name)
			if self.padding != 0 and not filedata.startswith((b'FLYT', b'FLAN')):  #Avoiding division by 0 + little hack
				file.write(bytes(self.padding - (file.tell() % self.padding or self.padding)))
			node.start = file.tell() - datastart
			file.write(filedata)
			pos = file.tell()
			node.end = pos - datastart
		pos = file.tell()
		file.write(bytes(0x80 - (file.tell() % 0x80 or 0x80)))
		endpad = file.tell() - pos
		if self.verbose:
			print('Editing references')
		filelen = file.tell() - endpad
		for node in self.nodes:
			file.seek(node.sfatnameoffset)
			self.pack('3I', node.nameoffset | (node.has_name << 24), node.start, node.end, file)
		file.seek(0)
		self.pack(SARC_HEADER_STRUCT, b'SARC', 20, 0xfeff, filelen, datastart, 0x00000100, file)
		if self.embedded:
			file.seek(0)
			return file.read(), endpad
		else:
			file.close()
	
	def calc_hash(self, name):
		result = 0
		for c in name:
			result = ord(c) + (result * self.hash_multiplier)
			result &= 0xFFFFFFFF
		return result
