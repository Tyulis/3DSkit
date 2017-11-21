# -*- coding:utf-8 -*-
import os
import sys
from io import BytesIO
from util import error


NEEDS_ENDIAN = (
	'GFA', 'BL'
)

USE_FILE_OBJ = (
	'GARC', 'BL', 'SARC', 'ALYT', 'BCSAR',
	'BFLIM', 'BFLAN',
)

MAGICS = {
	b'SARC': 'SARC',
	b'darc': 'DARC',
	b'CRAG': 'GARC',
	b'CBMD': 'CBMD',
	b'ALYT': 'ALYT',
	b'CLYT': 'BCLYT',
	b'FLYT': 'BFLYT',
	b'FLIM': 'BFLIM',
	b'FLAN': 'BFLAN',
	b'GFAC': 'GFA',
	b'CLIM': 'BCLIM',
	b'CSAR': 'BCSAR',
	b'CSTM': 'BCSTM',
	b'BL': 'BL',
	b'NCCH': 'NCCH',
	#Fake magics, avoid creating a list of supported formats (MAGICS.values() works)
	b'---a': 'ExtHeader',
	b'---b': 'ExeFS',
	b'---c': 'RomFS',
	b'---d': 'NDS',
}

EXTS = {
	'nds': 'NDS',
	'exefs': 'ExeFS',
	'romfs': 'RomFS',
}

SKIP_DECOMPRESSION = ('BFLIM', 'BCLIM', 'NCCH')


def recognize(filename, format=None):
	if format is not None:
		if format in MAGICS.values():
			return format
		else:
			error('Unsupported format to extract: %s. Read the formats section of the help for more infos.', 101)
	if hasattr(filename, 'read'):  #by get_ext
		file = filename
	else:
		if filename.lower() in ('extheader.bin', 'exheader.bin', 'decryptedexheader.bin'):
			return 'ExtHeader'
		if filename.lower() in ('exefs.bin', 'decryptedexefs.bin'):
			return 'ExeFS'
		if filename.lower() in ('romfs.bin', 'decryptedromfs.bin'):
			return 'RomFS'
		try:
			file = open(filename, 'rb')
		except OSError:
			error('File %s not found' % filename, 403)
	file.seek(0)
	magic = file.read(4)
	if magic in MAGICS.keys():
		return MAGICS[magic]
	if magic[0:2] in MAGICS.keys():
		return MAGICS[magic[0:2]]
	file.seek(-0x28, 2)
	magic = file.read(4)
	if magic in (b'FLIM', b'CLIM'):
		return MAGICS[magic]
	file.seek(0x100)
	magic = file.read(4)
	if magic == b'NCCH':
		return 'NCCH'
	if type(filename) == str:
		try:
			ext = os.path.split(filename)[-1].lower().split('.')[-1]
		except IndexError:
			return None
		for e in EXTS:
			if e in ext:
				return EXTS[e]
				sys.stdout.write('From extension: ')
	return None


def get_ext(data):
	if hasattr(data, 'read'):
		format = recognize(data)
		data.seek(0)
	else:
		format = recognize(BytesIO(data))
	if format is not None:
		return '.' + format.lower()
	else:
		return '.bin'
