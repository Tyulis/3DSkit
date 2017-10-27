# -*- coding:utf-8 -*-
formats = (
	'SARC',
	'DARC',
	'BFLYT',
	'ALYT',
	'BFLIM',
	'BL',
)

need_version = ('GARC',)
need_swizzle = ('BFLIM', 'BCLIM')


def pack(filenames, outname, format, endian, opts):
	print('Packing %s' % format)
	mod = __import__('pack.%s' % format)
	func = eval('mod.%s.pack%s' % (format, format))
	args = [filenames, outname, endian, opts]
	func(*args)
