# -*- coding:utf-8 -*-
import os
import sys
from io import BytesIO
from util import error


NEEDS_ENDIAN = (
	'GFA', 'mini'
)

USE_FILE_OBJ = (
	'GARC', 'mini', 'SARC', 'ALYT', 'BCSAR',
	'BFLIM', 'BFLAN', 'CGFX', 'NARC', 'NCCH',
	'ExeFS', 'RomFS', 'GFA', 'NDS', 'ExtHeader',
	'CBMD', 'BCSTM', 'BFFNT', 'CRO', 'MSBT',
	'BCGRP', 'BCWAR'
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
	b'CGFX': 'CGFX',
	b'GFAC': 'GFA',
	b'CLIM': 'BCLIM',
	b'CSAR': 'BCSAR',
	b'CSTM': 'BCSTM',
	b'NARC': 'NARC',
	b'NCCH': 'NCCH',
	b'FFNT': 'BFFNT',
	b'CGRP': 'BCGRP',
	b'CWAR': 'BCWAR',
	b'CWAV': 'BCWAV',
	b'CSEQ': 'BCSEQ',
	b'CBNK': 'BCBNK',
	#b'CRO0': 'CRO',
	b'MsgStdBn': 'MSBT',
	#Fake magics, avoid creating a list of supported formats (MAGICS.values() works)
	b'---a': 'ExtHeader',
	b'---b': 'ExeFS',
	b'---c': 'RomFS',
	#Mini known magics
	b'BL':   'mini',
	b'WD':   'mini',
	b'EM':   'mini',
}

EXTS = {
	'nds': 'NDS',
	'exefs': 'ExeFS',
	'romfs': 'RomFS',
}

SKIP_DECOMPRESSION = ('BFLIM', 'BCLIM', 'NCCH', 'CRO')


def recognize_filename(filename, format=None):
	filename = os.path.split(filename)[-1]
	if format is not None:
		return format
	if filename.lower() in ('extheader.bin', 'exheader.bin', 'exh.bin', 'decryptedextheader.bin'):
		return 'ExtHeader'
	elif filename.lower() in ('exefs.bin', 'decryptedexefs.bin'):
		return 'ExeFS'
	elif filename.lower() in ('romfs.bin', 'decryptedromfs.bin'):
		return 'RomFS'
	try:
		ext = os.path.splitext(filename)[-1].strip('.')
	except IndexError:
		ext = None
	if ext in EXTS:
		return EXTS[ext]
	return None

def recognize_file(file, format=None):
	if format is not None:
		return format
	file.seek(0, 2)
	filelen = file.tell()
	file.seek(0)
	if filelen >= 8:
		magic = file.read(8)
		if magic in MAGICS:
			return MAGICS[magic]
		if magic[0: 4] in MAGICS:
			return MAGICS[magic[0: 4]]
		if magic[0: 2] in MAGICS:
			return MAGICS[magic[0: 2]]
	if filelen >= 0x28:
		file.seek(-0x28, 2)
		magic = file.read(4)
		if magic in (b'FLIM', b'CLIM'):
			return MAGICS[magic]
	if filelen >= 0x84:
		file.seek(0x80)
		magic = file.read(4)
		if magic == b'CRO0':
			return 'CRO'
	if filelen >= 0x104:
		file.seek(0x100)
		magic = file.read(4)
		if magic == b'NCCH':
			return 'NCCH'
	return None


def get_ext(data):
	if hasattr(data, 'read'):
		format = recognize_file(data)
		data.seek(0)
	else:
		format = recognize_file(BytesIO(data))
	if format is not None:
		return '.' + format.lower()
	else:
		return '.bin'
