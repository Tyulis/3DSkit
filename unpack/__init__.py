# -*- coding:utf-8 -*-
from util.fileops import bread
from ._formats import *


def extract(filename, format, endian, verbose, opts, content=None):
	mod = __import__('unpack.%s' % format)
	cls = eval('mod.%s.extract%s' % (format, format))  #shorter than if format == 'SARC': extractSARC()...
	if format not in USE_FILE_OBJ:
		if content is not None:
			args = [filename, content, verbose]
		else:
			args = [filename, bread(filename), verbose]
	else:
		args = [filename, open(filename, 'rb'), verbose]
	if format in NEEDS_ENDIAN:
		args.append(endian)
	args.append(opts)
	unpacker = cls(*args)
	if unpacker is not None:  #if an archive
		unpacker.extract()


def list_files(filename, format, endian, verbose, opts, content=None):
	mod = __import__('unpack.%s' % format)
	cls = eval('mod.%s.extract%s' % (format, format))  #shorter than if magic == 'SARC': extractSARC()...
	if format not in USE_FILE_OBJ:
		if content is not None:
			args = [filename, content, verbose]
		else:
			args = [filename, bread(filename), verbose]
	else:
		args = [filename, open(filename, 'rb'), verbose]
	if format in need_endian:
		args.append(endian)
	args.append(opts)
	unpacker = cls(*args)
	unpacker.list()
