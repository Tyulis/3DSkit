# -*- coding:utf-8 -*-
from ._formats import *

def pack(filenames, outname, format, endian, verbose, opts):
	print('Packing %s' % format)
	mod = __import__('pack.%s' % format)
	func = eval('mod.%s.pack%s' % (format, format))
	args = [filenames, outname, endian, verbose, opts]
	func(*args)
