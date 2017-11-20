import sys

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
	sys.exit(errno)

def warning(msg, errno):
	print('Warning: %s (%d)' % (msg, errno))
	return errno
