from util.fileops import bread
from ._formats import *


def extract(content, filename, format, endian):
	mod = __import__('unpack.%s' % format)
	cls = eval('mod.%s.extract%s' % (format, format))  #shorter than if magic == 'SARC': extractSARC()...
	args = [filename, content]
	if format in need_endian:
		args.append(endian)
	unpacker = cls(*args)
	if unpacker is not None:  #if not an archive
		unpacker.extract()


def list_files(content, filename, format, endian):
	mod = __import__('unpack.%s' % format)
	cls = eval('mod.%s.extract%s' % (format, format))  #shorter than if magic == 'SARC': extractSARC()...
	args = [filename, content]
	if format in need_endian:
		args.append(endian)
	unpacker = cls(*args)
	unpacker.list()
