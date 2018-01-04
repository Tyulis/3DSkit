# -*- coding:utf-8 -*-
from util import error

def recognize(file, recursive_mode=False):
	magic = file.read(4)
	if len(magic) == 0:
		return 0
	if magic[0] == 0x10:
		return 'LZ10'
	elif magic[0] == 0x11:
		return 'LZ11'
	elif magic[0] == 0x40:
		return 'LZH8'
	elif magic == b'Yaz0':
		return 'Yaz0'
	return None


def decompress(file, out, format, verbose, errorcb=lambda e: error.InternalCorrectionWarning('Bad detection of the compression')):
	file.seek(0)
	out.seek(0)
	mod = __import__('compress.%s' % format)
	func = eval('mod.%s.decompress%s' % (format, format))
	if 1:
		func(file, out, verbose)
	else:  #except Exception as e:  #Bad detection
		return errorcb(e)


def compress(file, out, format, verbose):
	file.seek(0)
	out.seek(0)
	mod = __import__('compress.%s' % format)
	func = eval('mod.%s.compress%s' % (format, format))
	func(file, out, verbose)
