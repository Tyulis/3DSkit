def recognize(cnt):
	if cnt[0] == 0x10:
		return 'LZ10'
	elif cnt[0] == 0x11:
		return 'LZ11'
	elif cnt[0] == 0x40:
		return 'LZH8'
	elif cnt[0:4] == b'Yaz0':
		return 'Yaz0'
	return None


def decompress(cnt, format):
	mod = __import__('compress.%s' % format)
	func = eval('mod.%s.decompress%s' % (format, format))
	return func(cnt)


def compress(cnt, format):
	mod = __import__('compress.%s' % format)
	func = eval('mod.%s.compress%s' % (format, format))
	return func(cnt)
