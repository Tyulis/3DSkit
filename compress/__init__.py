DEC_USE_FILE_OBJS = (
	'Yaz0',
	'LZ11',
)

CMP_USE_FILE_OBJS = (
	'LZ11',
)


def recognize(file):
	magic = file.read(4)
	if magic[0] == 0x10:
		return 'LZ10'
	elif magic[0] == 0x11:
		return 'LZ11'
	elif magic[0] == 0x40:
		return 'LZH8'
	elif magic == b'Yaz0':
		return 'Yaz0'
	return None


def decompress(file, out, format, verbose):
	file.seek(0)
	out.seek(0)
	mod = __import__('compress.%s' % format)
	func = eval('mod.%s.decompress%s' % (format, format))
	if format in DEC_USE_FILE_OBJS:
		func(file, out, verbose)
	else:
		final = func(file.read(), verbose)
		out.write(final)


def compress(file, out, format, verbose):
	file.seek(0)
	out.seek(0)
	mod = __import__('compress.%s' % format)
	func = eval('mod.%s.compress%s' % (format, format))
	if format in CMP_USE_FILE_OBJS:
		func(file, out, verbose)
	else:
		final = func(file.read(), verbose)
		out.write(final)
