# -*- coding:utf-8 -*-
#A little and ugly script I use to see the differences between 2 binary files (e.g between an original and a packed file)
import console
from ppy import *

a=bread('../../test/0.alyt')
b=bread('../../test.alyt')

i=0
if a == b:
	print('No issues found')
	exit()
while True:
	aa = a[i: i + 4]
	bb=b[i:i+4]
	ahex=aa.hex()
	aascii=''.join([(chr(c) if c in range (32,128) else '.') for c in aa])
	bhex=bb.hex()
	bascii=''.join([(chr(c) if c in range (32,128) else '.') for c in bb])
	ahex=' '*(8-len(ahex))+ahex
	aascii=' '*(4-len(aascii))+aascii
	bhex=' '*(8-len(bhex))+bhex
	bascii=' '*(4-len(bascii))+bascii
	if ahex!=bhex:
		console.set_color(1.0,0.2,0.2)
	else:
		console.set_color(1.0, 0.77, 0.33)
	if ahex != bhex:
		print('%08x %s %s %s %s'%(i, ahex,bhex,aascii,bascii))
	if i>max((len(a),len(b))):
		break
	i+=4

console.set_color(1.0, 0.77, 0.33)
