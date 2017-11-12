# -*- coding:utf-8 -*-
import os
import sys
from util import error


NEEDS_ENDIAN = (
	'GFA', 'BL'
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
	b'GFAC': 'GFA',
	b'CLIM': 'BCLIM',
	b'CSAR': 'BCSAR',
	b'CSTM': 'BCSTM',
	b'BL':   'BL',
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


def recognize(cnt, filename='', format=None):
	if format is not None:
		if format in MAGICS.values():
			return format
		else:
			error('Unsupported format to extract: %s. Read the formats section of the help for more infos.', 101)
	if filename.lower() in ('extheader.bin', 'exheader.bin', 'decryptedexheader.bin'):
		return 'ExtHeader'
	if filename.lower() in ('exefs.bin', 'decryptedexefs.bin'):
		return 'ExeFS'
	if filename.lower() in ('romfs.bin', 'decryptedromfs.bin'):
		return 'RomFS'
	if len(cnt) >= 4:
		if cnt[0:4] in MAGICS.keys():
			return MAGICS[cnt[0:4]]
	if len(cnt) >= 2:
		if cnt[0:2] in MAGICS.keys():
			return MAGICS[cnt[0:2]]
	if len(cnt) >= 0x28:
		if cnt[-0x28:-0x24] in (b'FLIM', b'CLIM'):
			return MAGICS[cnt[-0x28:-0x24]]
	if len(cnt) >= 0x104:
		if cnt[0x100: 0x104] == b'NCCH':
			return 'NCCH'
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
	format = recognize(data)
	if format is not None:
		return '.' + format.lower()
	else:
		return '.bin'
