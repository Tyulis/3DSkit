import sys
import math
from .funcops import ClsFunc

BOMS = {
	'>': 0xfeff,
	'<': 0xfffe
}

ENDIANS = {
	0xfeff: '>',
	0xfffe: '<'
}


class error (ClsFunc):
	unsupportedformat_unpack = 101
	unsupportedformat_pack = 102
	unrecognizedformat = 103
	uncompressedfile = 104
	unsupported_colorformat = 105
	unsupported_version = 106
	unsupported_setting = 107
	unsupported_coding = 108
	pluginnotfound = 207
	invalidmagic = 301
	invalidsection_pack = 302
	invalidsection_unpack = 303
	hashmismatch = 305
	invalidformat = 306
	def main(self, msg, errno):
		if math.floor(errno / 100) != 9:
			print('Error: %s (%d)' % (msg, errno))
			sys.exit(errno)
		else:
			print('Warning: %s (%d)' % (msg, errno))
			return errno
