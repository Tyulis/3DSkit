#A little script I made to bruteforce the hash algorithm and multiplier in GFAC files
n = b'BattleUp.arc'
h = 0xf86c352d
m = 0xffffffff

for mul in range(0, 65536):
	r = 0
	for c in n:
		r *= mul
		r += c
		r &= m
	if r == h:
		print('Multiplier: %04x' % mul)
		break
else:
	print('No multiplier found')
