formats=(
	'SARC',
	'DARC',
	'GARC',
	'BFLYT',
	'ALYT',
)

need_version = ('GARC',)
need_swizzle = ('BFLIM', 'BCLIM')

def pack(filenames, outname, format, endian, version, swizzle):
	print('Packing %s' % format)
	mod = __import__('pack.%s' % format)
	func = eval('mod.%s.pack%s' % (format, format))
	args = [filenames, outname, endian]
	if format in need_version:
		args.append(version)
	if format in need_swizzle:
		args.append(swizzle)
	func(*args)
