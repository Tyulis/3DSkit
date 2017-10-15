# -*- coding:utf-8 -*-
import os
from util import error
from util.fileops import *
import util.rawutil as rawutil

GFAC_META_STRUCT = '4s11I/11[4I]/11[n]128a'
GFCP_HEADER_STRUCT = '4s4I'
GFAC_HASH_MULTIPLIER = 0x89


class GFA_FATnode (object):
	pass


class extractGFA (rawutil.TypeReader):
	def __init__(self, filename, data, endian, opts={}):
		self.byteorder = endian
		self.outdir = make_outdir(filename)
		ptr = self.readmeta(data)
		self.readinfo()
		self.gfcp = data[ptr:]
	
	def calc_hash(self, name):
		hash = 0
		for c in name:
			hash *= GFAC_HASH_MULTIPLIER
			hash += c
			hash &= 0xffffffff
		return hash
	
	def readmeta(self, data):
		hdata, ptr = self.unpack_from(GFAC_META_STRUCT, data, 0, getptr=True)
		magic = hdata[0]
		if magic != b'GFAC':
			error('Invalid magic: %s' % magic)
		#unknown = hdata[1]
		self.version = hdata[2]
		#headerlen = hdata[3]
		#metalen = hdata[4]
		self.dataoffset = hdata[5]
		#datalen = hdata[6]
		self.filecount = hdata[11]
		self.fat = hdata[12]
		self.filenames = hdata[13]
		self.readinfo()
		return ptr
	
	def readinfo(self):
		self.nodes = [self.makenode(self.fat[i], self.filenames[i][0]) for i in range(0, self.filecount)]
	
	def makenode(self, entry, name):
		node = GFA_FATnode()
		node.name = name.decode('utf-8')
		node.hash = entry[0]
		hash = self.calc_hash(name)
		if hash != node.hash:
			error('Invalid file name hash for %s (found %08x, expected %08x)' % (node.name, node.hash, hash))
		# node.nameoffset  = entry[1]
		node.length = entry[2]
		node.offset = entry[3] - self.dataoffset
		return node
	
	def list(self):
		for node in self.nodes:
			print(node.name)
	
	def extract(self):
		data = self.gfcp
		magic, version, comp, self.decsize, compsize = self.unpack_from(GFCP_HEADER_STRUCT, data)
		data = data[20:]
		if magic != b'GFCP':
			error('Invalid GFCP magic: %s' % magic)
		if comp in (2, 3):
			print('\tDecompressing data...')
			data = self.decompressLZ10(data)
			print('\tOK!')
		elif comp == 1:
			error('Actually unsupported compression')
		else:
			error('Invalid compression')
		for node in self.nodes:
			outname = self.outdir + node.name
			filedata = data[node.offset:node.offset + node.length]
			bwrite(filedata, outname)
	
	def decompressLZ10(self, data):
		# cannot use compress.LZ10 because of strange header
		ptr = 0
		final = []
		while len(final) < self.decsize:
			flags = self.tobits(data[ptr])
			ptr += 1
			for flag in flags:
				if flag == 0:
					byte, ptr = self.uint8(data, ptr)
					final.append(byte)
				else:
					infobs = rawutil.unpack_from('>H', data, ptr)[0]
					ptr += 2
					count = (infobs >> 12) + 3
					disp = (infobs & 0xfff) + 1
					for i in range(0, count):
						final.append(final[-disp])
				if len(final) >= self.decsize:
					break
		ret = bytes(final)
		return ret
