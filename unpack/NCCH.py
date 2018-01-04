# -*- coding:utf-8 -*-
from util import error
from util.utils import byterepr
from util.filesystem import *
import util.rawutil as rawutil
import util.txtree as txtree
from collections import OrderedDict
from hashlib import sha256
from io import BytesIO

from compress.LZ11 import decompressLZ11
from unpack.ExtHeader import extractExtHeader
from unpack.ExeFS import extractExeFS
from unpack.RomFS import extractRomFS
from unpack.DARC import extractDARC

NCCH_HEADER_STRUCT = '256X 4s IQ2HIQ 16X 32s n16a 32s 2I (8B) 12I 32s32s'

CRYPTO_KEYSLOTS = {
	0x00: None,
	0x01: 0x25,
	0x0a: 0x18,
	0x0b: 0x1b,
}

PLATFORMS = {
	1: 'CTR (o3DS)',
	2: 'SNAKE (n3DS)'
}
	

class extractNCCH (rawutil.TypeReader):
	def __init__(self, filename, data, verbose, opts={}):
		self.verbose = verbose
		self.dochecks = True
		self.dumpfs = False
		if 'dochecks' in opts.keys():
			self.dochecks = True if opts['dochecks'].lower() == 'true' else False
		if 'dumpfs' in opts.keys():
			self.dumpfs = True if opts['dumpfs'].lower() == 'true' else False
		self.outdir = make_outdir(filename)
		self.byteorder = '<'
		self.data = data
		self.readheader()
		self.convert_units()
	
	def readheader(self,):
		data = self.unpack_from(NCCH_HEADER_STRUCT, self.data, 0)
		self.rsa_signature = data[0]
		magic = data[1]
		if magic != b'NCCH':
			error.InvalidMagicError('Invalid magic %s, expected NCCH' % byterepr(magic))
		self.content_size = data[2]  #In media units
		self.partition_id = rawutil.hex(data[3], 16)
		self.maker_code = data[4]
		self.version = data[5]
		self.id_hash = data[6]
		self.tid_high = rawutil.hex(data[7] >> 32, 8)
		self.tid_low = rawutil.hex(data[7] & 0xffffffff, 8)
		#reserved = data[8]
		self.logo_hash = data[9]
		self.product_code = data[10].decode('ascii')
		self.extheader_hash = data[11]
		self.extheader_size = data[12]  #in bytes
		#reserved = data[13]
		ncchflags = data[14]
		self.crypto_keyslot = CRYPTO_KEYSLOTS[ncchflags[3]]
		self.platform = PLATFORMS[ncchflags[4]]
		self.content_types = self.extract_contenttypes(ncchflags[5])
		self.unitsize = 0x200 << ncchflags[6]
		flags = ncchflags[7]
		self.fixed_cryptokey = bool(flags & 0x01)
		self.has_romfs = not bool(flags & 0x02)
		self.has_crypto = not bool(flags & 0x04)
		self.use_new_keyYgen = bool(flags & 0x20)
		#Everything is in media units
		self.plain_offset = data[15]
		self.plain_size = data[16]
		self.logo_offset = data[17]
		self.logo_size = data[18]
		self.exefs_offset = data[19]
		self.exefs_size = data[20]
		self.exefs_hashregion_size = data[21]
		#reserved = data[22]
		self.romfs_offset = data[23]
		self.romfs_size = data[24]
		self.romfs_hashregion_size = data[25]
		#reserved = data[26]
		self.exefs_hash = data[27]
		self.romfs_hash = data[28]
	
	def extract_contenttypes(self, flags):
		types = []
		if flags & 0x01:
			types.append('Data')
		elif flags & 0x02:
			types.append('Executable')
		elif flags & 0x04 and not flags & 0x08:
			types.append('System Update')
		elif flags & 0x08 and not flags & 0x04:
			types.append('Manual')
		elif flags & 0x0c:
			types.append('Child')
		elif flags & 0x10:
			types.append('Trial')
		return types
	
	def convert_units(self):
		self.content_size *= self.unitsize
		self.logo_size *= self.unitsize
		self.exefs_size *= self.unitsize
		self.romfs_size *= self.unitsize
		self.plain_size *= self.unitsize
		self.exefs_hashregion_size *= self.unitsize
		self.romfs_hashregion_size *= self.unitsize
		self.logo_offset *= self.unitsize
		self.plain_offset *= self.unitsize
		self.exefs_offset *= self.unitsize
		self.romfs_offset *= self.unitsize

	def extract(self):
		#Code to really extract files
		self.extract_data()
		self.extract_sections()
		self.extract_subs()
	
	def extract_data(self):
		tree = OrderedDict()
		tree['NCCH'] = OrderedDict()
		root = tree['NCCH']
		root['PartitionID'] = self.partition_id, 16
		root['TitleID'] = {'High': self.tid_high, 'Low ': self.tid_low}
		root['Product-code'] = self.product_code
		root['Version'] = self.version
		root['Makercode'] = self.maker_code
		root['Platform'] = self.platform
		root['Content-types'] = self.content_types
		root['Has-RomFS'] = self.has_romfs
		root['Has-Crypto'] = self.has_crypto
		root['Fixed-Crypto-key'] = self.fixed_cryptokey
		root['Crypto-Keyslot'] = self.crypto_keyslot
		root['Uses-new-KeyY-generator'] = self.use_new_keyYgen
		root['RSA-signature'] = self.rsa_signature
		root['Hashes'] = OrderedDict()
		hashes = root['Hashes']
		hashes['ExeFS'] = rawutil.hex(self.exefs_hash)
		hashes['RomFS'] = rawutil.hex(self.romfs_hash)
		hashes['Logo'] = rawutil.hex(self.logo_hash)
		hashes['ExtHeader'] = rawutil.hex(self.extheader_hash)
		hashes['ID'] = rawutil.hex(self.id_hash, 8)
		root['Structure'] = OrderedDict()
		struct = root['Structure']
		struct['Media-size-unit'] = self.unitsize
		struct['Comment'] = 'All these sizes and offsets are in bytes'
		struct['ExtHeader-size'] = self.extheader_size
		struct['Plain-offset'] = self.plain_offset
		struct['Plain-size'] = self.plain_size
		struct['Logo-offset'] = self.logo_offset
		struct['Logo-size'] = self.logo_size
		struct['ExeFS-offset'] = self.exefs_offset
		struct['ExeFS-size'] = self.exefs_size
		struct['ExeFS-HashRegion-size'] = self.exefs_hashregion_size
		struct['RomFS-offset'] = self.romfs_offset
		struct['RomFS-size'] = self.romfs_size
		struct['RomFS-HashRegion-size'] = self.romfs_hashregion_size
		final = txtree.dump(tree)
		write(final, self.outdir + 'header.txt')
	
	def extract_sections(self):
		self.data.seek(0x200)
		extheader = self.data.read(self.extheader_size * 2)
		self.data.seek(self.plain_offset)
		plain = self.data.read(self.plain_size)
		self.data.seek(self.logo_offset)
		logo = self.data.read(self.logo_size)
		#exefs = self.data[self.exefs_offset: self.exefs_offset + self.exefs_size]
		#romfs = self.data[self.romfs_offset: self.romfs_offset + self.romfs_size]
		if len(logo) > 0:
			self.has_logo = True
		else:
			self.has_logo = False
		if self.dochecks:
			self.data.seek(0x200)
			if sha256(self.data.read(self.extheader_size)).digest() != self.extheader_hash:
				error.HashMismatchError('Extended header hash mismatch')
			if sha256(logo).digest() != self.logo_hash and logo != b'':
				error.HashMismatchError('Logo hash mismatch')
			self.data.seek(self.exefs_offset)
			if sha256(self.data.read(self.exefs_hashregion_size)).digest() != self.exefs_hash:
				error.HashMismatchError('ExeFS hash mismatch')
			if self.romfs_hashregion_size < (2 ** 20) and self.has_romfs:
				self.data.seek(self.romfs_offset)
				if sha256(self.data.read(self.romfs_hashregion_size)).digest() != self.romfs_hash:
					error.HashMismatchError('RomFS hash mismatch')
		bwrite(extheader, self.outdir + 'extheader.bin')
		if len(plain) > 0:
			bwrite(plain, self.outdir + 'plain.bin')
		if len(logo) > 0:
			bwrite(logo, self.outdir + 'logo.darc')
		if self.dumpfs:
			with open(self.outdir + 'exefs.bin', 'wb') as exefs_out:
				self.data.seek(self.exefs_offset)
				exefs_out.write(self.data.read(self.exefs_size))
			if self.has_romfs:
				with open(self.outdir + 'romfs.bin', 'wb') as romfs_out:
					self.data.seek(self.romfs_offset)
					romfs_out.write(self.data.read(self.romfs_size))

	def extract_subs(self):
		if self.has_logo:
			logopath = self.outdir + 'logo.darc'
			self.data.seek(self.logo_offset)
			input = BytesIO(self.data.read(self.logo_size))
			output = BytesIO()
			decompressLZ11(input, output, self.verbose)
			output.seek(0)
			logo_extractor = extractDARC(logopath, output.read(), self.verbose)
			logo_extractor.extract()
		extheaderpath = self.outdir + 'extheader.bin'
		with open(extheaderpath, 'rb') as exh:
			extractExtHeader(extheaderpath, exh, self.verbose)
		exefspath = self.outdir + 'exefs.bin'
		#Not so huge -- we keep this for the moment
		self.data.seek(self.exefs_offset)
		exefs_extractor = extractExeFS(exefspath, BytesIO(self.data.read(self.exefs_size)), self.verbose, opts={'dochecks': str(self.dochecks)})
		exefs_extractor.extract()
		if self.has_romfs:
			#Keep a reference into the NCCH partition
			#Because a RomFS can take up to 3GB, we cannot map it into memory with a BytesIO
			#Let's do that in a more hacky (and fast) way
			romfspath = self.outdir + 'romfs.bin'
			opts = {'dochecks': str(self.dochecks), 'baseoffset': self.romfs_offset}
			romfs_extractor = extractRomFS(romfspath, self.data, self.verbose, opts=opts)
			romfs_extractor.extract()
