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
	print('Error: %s (%d)' % (msg, errno))
	if math.floor(errno / 100) != 9:
		sys.exit(errno)
