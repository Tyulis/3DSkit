import sys
import math

BOMS = {
	'>': 0xfeff,
	'<': 0xfffe
}

ENDIANS = {
	0xfeff: '>',
	0xfffe: '<'
}


def error(msg, errno):
	if math.floor(errno / 100) != 9:
		print('Error: %s (%d)' % (msg, errno))
		sys.exit(errno)
	else:
		print('Warning: %s (%d)' % (msg, errno))
		return errno
