# -*- coding:utf-8 -*-
import os
from util import error
from util.utils import byterepr
from util.filesystem import *
import util.rawutil as rawutil

IVFC_HEADER_STRUCT = '4s2I 3[2Q2I] 2I'
IVFC_HEADER_NAMES = 'magic, version, masterhash_size, levels, reserved, optinfo_size'
IVFC_LEVEL3_HEADER_STRUCT = 'I 4[2I] I'
IVFC_LEVEL3_HEADER_NAMES = 'headerlen, tables, filedata_offset'
IVFC_DIR_METADATA_STRUCT = '6I /p1s 4a'
IVFC_FILE_METADATA_STRUCT = '2I2Q2I /p1s 4a'


class DirectoryEntry (object):
	def __init__(self, data):
		self.children = []
		self.parent_offset, self.next_offset, self.child_offset, self.firstfile_offset, self.next_bucket_dir_offset, namelen, self.name = data
		self.name = self.name.decode('utf-16-le')

class FileEntry (object):
	def __init__(self, data):
		self.parentdir_offset, self.next_offset, self.data_offset, self.data_size, self.next_bucket_file_offset, namelen, name = data
		self.name = name.decode('utf-16-le')


class extractRomFS (rawutil.TypeReader):
	def __init__(self, filename, data, verbose, opts={}):
		self.outdir = make_outdir(filename)
		self.verbose = verbose
		self.byteorder = '<'
		self.dochecks = False
		self.base = 0
		if 'dochecks' in opts:
			self.dochecks = True if opts['dochecks'].lower() == 'true' else False
		if 'baseoffset' in opts:
			#To read the RomFS directly into the NCCH file (see unpack/NCCH for info)
			self.base = int(opts['baseoffset'])
		self.data = data
		self.read_ivfc_tree()
	
	def read_ivfc_tree(self):
		header = self.unpack_from(IVFC_HEADER_STRUCT, self.data, self.base, IVFC_HEADER_NAMES)
		if header.magic != b'IVFC':
			error.InvalidMagicError('Invalid magic, %s, expected IVFC' % byterepr(header.magic))
		#self.read_level3(*header.levels[2][:3])
		self.read_level3(0x1000, *header.levels[2][1:3])
	
	def read_level3(self, offset, size, blocklen):
		self.level3_offset = self.base + offset
		self.blocklen = 2 ** blocklen
		header = self.unpack_from(IVFC_LEVEL3_HEADER_STRUCT, self.data, self.level3_offset, IVFC_LEVEL3_HEADER_NAMES)
		self.read_level3_tree(header.tables[1][0], header.tables[3][0])
		self.data_offset = self.level3_offset + header.filedata_offset
	
	def read_level3_tree(self, diroffset, fileoffset):
		self.dirmeta_offset = self.level3_offset + diroffset
		self.filemeta_offset = self.level3_offset + fileoffset
		self.root = DirectoryEntry(self.unpack_from(IVFC_DIR_METADATA_STRUCT, self.data, self.dirmeta_offset))
		#self.unpack_from()
		if self.root.firstfile_offset != 0xffffffff:
			self.read_subfile_meta(self.root)
		if self.root.child_offset != 0xffffffff:
			self.read_subdir_meta(self.root)
	
	def read_subdir_meta(self, dir):
		subdir = DirectoryEntry(self.unpack_from(IVFC_DIR_METADATA_STRUCT, self.data, dir.child_offset + self.dirmeta_offset))
		if subdir.firstfile_offset != 0xffffffff:
			self.read_subfile_meta(subdir)
		if subdir.child_offset != 0xffffffff:
			self.read_subdir_meta(subdir)
		dir.children.append(subdir)
		while subdir.next_offset != 0xffffffff:
			subdir = DirectoryEntry(self.unpack_from(IVFC_DIR_METADATA_STRUCT, self.data, subdir.next_offset + self.dirmeta_offset))
			if subdir.firstfile_offset != 0xffffffff:
				self.read_subfile_meta(subdir)
			if subdir.child_offset != 0xffffffff:
				self.read_subdir_meta(subdir)
			dir.children.append(subdir)
	
	def read_subfile_meta(self, dir):
		file = FileEntry(self.unpack_from(IVFC_FILE_METADATA_STRUCT, self.data, dir.firstfile_offset + self.filemeta_offset))
		dir.children.append(file)
		while file.next_offset != 0xffffffff:
			file = FileEntry(self.unpack_from(IVFC_FILE_METADATA_STRUCT, self.data, file.next_offset + self.filemeta_offset))
			dir.children.append(file)

	def extract(self):
		#Code to really extract files
		self.root.path = self.outdir
		self.extract_dir(self.root)
	
	def extract_dir(self, dir):
		mkdir(dir.path)
		for child in dir.children:
			if isinstance(child, DirectoryEntry):
				child.path = dir.path + child.name + os.path.sep
				self.extract_dir(child)
			else:
				path = dir.path + child.name
				self.data.seek(self.data_offset + child.data_offset)
				with open(path, 'wb') as file:
					file.write(self.data.read(child.data_size))
