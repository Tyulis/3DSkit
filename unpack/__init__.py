# -*- coding:utf-8 -*-
from ._formats import *


def extract(filename, file, format, endian, verbose, opts):
	mod = __import__('unpack.%s' % format)
	cls = eval('mod.%s.extract%s' % (format, format))  #shorter than if format == 'SARC': extractSARC()...
	file.seek(0)
	if format not in USE_FILE_OBJ:
		args = [filename, file.read(), verbose]
	else:
		args = [filename, file, verbose]
	if format in NEEDS_ENDIAN:
		args.append(endian)
	args.append(opts)
	unpacker = cls(*args)
	if unpacker is not None:  #if an archive
		unpacker.extract()
