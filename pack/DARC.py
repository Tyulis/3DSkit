# -*- coding:utf-8 -*-
import os
from util.rawutil import TypeWriter
from util.fileops import bread, bwrite, basedir
from util.funcops import ClsFunc
from util import error, BOMS

DARC_HEADER_STRUCT = '4s4H4I'
DARC_TABLE_STRUCT = '#0[3I]'


class packDARC (ClsFunc, TypeWriter):
	def main(self, filenames, outname, endian, opts={}):
		self.byteorder = endian
		self.outname = outname
		tree = self.make_tree(filenames)
		final = self.pack_sections(tree)
		final = self.pack_header(final)
		basedir()
		bwrite(final, outname)
	
	def pack_header(self, final):
		hdr = self.pack(DARC_HEADER_STRUCT, b'darc', 0xfeff, 0x1c, 0, 0x100, len(final) + 28, 0x1c, len(self.table + self.names), self.metalen)
		return hdr + final
	
	def make_tree(self, filenames):
		tree = {'.':{}}
		for name in filenames:
			pointed = tree['.']
			path = name.split(os.path.sep)
			for i, el in enumerate(path):
				if i == len(path) - 1:  #file
					pointed[el] = bread(name)
				elif el in pointed.keys():  #existing folder
					pointed = pointed[el]
				else:  #create the folder
					pointed[el] = {}
					pointed = pointed[el]
		return tree
	
	def _getentryinfo(self, tree):
		for el in tree.keys():
			if isinstance(tree[el], dict):
				self._getentryinfo(tree[el])
			self.entrynum += 1
			self.entrynames.append(el)
	
	def getentryinfo(self, tree):
		self.entrynum = 1
		self.entrynames = ['']
		self._getentryinfo(tree)
		return self.entrynum, self.entrynames
	
	def pack_sections(self, tree):
		self.table = b''
		self.names = b'\x00\x00'
		self.files = b''
		num, names = self.getentryinfo(tree)
		namelen = 0
		for name in names:
			namelen += len(self.utf16string(name))
		self.metalen = 0x28 + 3 * num + namelen
		self.pack_folder(tree)
		self.table = self.pack('3I', 0x01000000, 0, (len(self.table) // 3) + 1) + self.table
		final = self.table + self.names + self.files
		return final
	
	def pack_folder(self, tree):
		for el in tree.keys():
			if isinstance(tree[el], dict):
				nameoffset = len(self.names) | 0x01000000
				self.names += self.utf16string(el)
				actentrynum = len(self.table) // 3
				endentry = actentrynum + len(tree[el]) - 1
				self.table += self.pack('3I', nameoffset, 1, endentry)
				self.pack_folder(tree[el])
			else:
				nameoffset = len(self.names)
				self.names += self.utf16string(el)
				dataoffset = len(self.files) + self.metalen
				datalength = len(tree[el])
				self.table += self.pack('3I', nameoffset, dataoffset, datalength)
				self.files += tree[el]
				self.files += self.align(self.files, 0x20)
