# -*- coding:utf-8 -*-
# An old and ugly script I use to see the differences between 2 binary files (e.g between an original and a packed file)

import os
import sys


def main(name1, name2):
	file1 = open(name1, 'rb')
	file2 = open(name2, 'rb')
	running = True
	pos = 0
	while True:
		data1 = file1.read(16)
		data2 = file2.read(16)
		if len(data1) == len(data2) == 0:
			break
		if data1 != data2:
			print('%08X : ' % pos, end='')
			result1 = ""
			result2 = ""
			for i in range(4):
				sub1 = data1[i * 4: (i + 1) * 4]
				sub2 = data2[i * 4: (i + 1) * 4]
				for j in range(4):
					byte1 = sub1[j]
					byte2 = sub2[j]
					if byte1 != byte2:
						result1 += '\033[91m%02x\033[0m' % byte1
						result2 += '\033[91m%02x\033[0m' % byte2
					else:
						result1 += '%02x' % byte1
						result2 += '%02x' % byte2
				result1 += ' '
				result2 += ' '
			print('%s | %s' % (result1, result2))
		pos += 16


if __name__ == '__main__':
	if len(sys.argv) != 3:
		print('Usage : python3 delta.py <file 1> <file 2>')
	main(sys.argv[1], sys.argv[2])
