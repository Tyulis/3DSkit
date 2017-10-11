import sys

BOMS = {
	'>': 0xfeff,
	'<': 0xfffe
}

ENDIANS = {
	0xfeff: '>',
	0xfffe: '<'
}

def error(msg):
	print(msg)
	sys.exit(1)
