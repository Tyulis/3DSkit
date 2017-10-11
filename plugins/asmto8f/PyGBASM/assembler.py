# -*- coding:utf-8 -*-
from ._intern import *

OPS_1B = (
	'nop', 'inc', 'dec', 'rrca', 'rra', 'cpl',
	'ccf', 'halt', 'ei', 'di', 'pop', 'push',
	'ret', 'reti', 'rst', 'rlca', 'rla', 'daa',
	'scf', 'ldi', 'ldd'
)
OPS_2B = (
	'stop', 'jr', 'rlc', 'rrc', 'rl', 'rr',
	'sla', 'sra', 'swap', 'srl', 'bit', 'res',
	'set', 'ldh', 'ldhl'
)
OPS_3B = (
	'call'
)
NUM_OPS = (
	'add', 'sub', 'adc', 'sbc', 'and', 'or', 'xor', 'cp'
)
REG_INDICS = {
	'b': 0, 'c': 1, 'd': 2, 'e': 3,
	'h': 4, 'l': 5, '(hl)': 6, 'a': 7
}

def _assemble(code, offset, labels):
	comp = b''
	for i, ln in enumerate(code):
		if ln.endswith(':'):  #labels
			continue
		l = lsplit(ln)
		op = l[0]
		if len(l) > 1:
			args = [el.strip() for el in l[1].split(',')]
		else:
			args = ['', '']  #uglily avoid errors
		if len(args) != 2:
			args.append('')
		if op == '.incbin':
			comp += bread(args[0].strip('\'"'))
		elif op == '.db':
			for n in args:
				comp += u8(toint(n))
		elif op == 'nop':  #nop: 00
			comp += u8(0)
		elif op == 'stop':  #stop: 1000
			comp += u16(0x1000)
		elif op == 'halt':  #halt: 76
			comp += u8(0x76)
		elif op == 'rlca':  #rlca: 07
			comp += u8(0x07)
		elif op == 'rla':  #rla: 17
			comp += u8(0x17)
		elif op == 'daa':  #daa: 27
			comp += u8(0x27)
		elif op == 'scf':  #scf: 37
			comp += u8(0x37)
		elif op == 'rrca':  #rrca: 0f
			comp += u8(0x0f)
		elif op == 'rra':  #rra: 1f
			comp += u8(0x1f)
		elif op == 'cpl':  #cpl: 2f
			comp += u8(0x2f)
		elif op == 'ccf':  #cpf: 3f
			comp += u8(0x3f)
		elif op == 'di':  #di: f3
			comp += u8(0xf3)
		elif op == 'ei':  #ei: fb
			comp += u8(0xfb)
		elif op == 'reti':  #reti: d9
			comp += u8(0xd9)
		elif op == 'rst':
			num = toint(args[0])
			comp += u8(0xc7 + num)
		elif op == 'jr':
			if args[1] in labels.keys():
				jump = labels[args[1]] - (getsize(code[:i]) + offset + 2)
				if abs(jump) > 127:
					error('Opcode %d: %s: Jump is over +-127 to label %s' % (i, ln, args[1]))
				args[1] = jump
			if args[0] == 'nz':  #jr nz,xx: 20xx
				comp += u8(0x20) + i8(args[1])
			elif args[0] == 'nc':  #jr nc,xx: 30xx
				comp += u8(0x30) + i8(args[1])
			elif args[0] == 'z':  #jr z,xx: 28xx
				comp += u8(0x28) + i8(args[1])
			elif args[0] == 'c':  #jr c,xx: 38xx
				comp += u8(0x38) + i8(args[1])
			elif args[0] == 'nn':  #jr xx: 18xx
				comp += u8(0x18) + i8(args[1])
			else:
				error('Opcode %d: %s: Unknown condition %s' % (i, ln, args[0]))
		elif op == 'ld':
			if args[0] == 'b':
				if args[1] == 'b':  #ld b,b: 40
					comp += u8(0x40)
				elif args[1] == 'c':  #ld b,c: 41
					comp += u8(0x41)
				elif args[1] == 'd':  #ld b,d: 42
					comp += u8(0x42)
				elif args[1] == 'e':  #ld b,e: 43
					comp += u8(0x43)
				elif args[1] == 'h':  #ld b,h: 44
					comp += u8(0x44)
				elif args[1] == 'l':  #ld b,l: 45
					comp += u8(0x45)
				elif args[1] == '(hl)':  #ld b,(hl): 46
					comp += u8(0x46)
				elif args[1] == 'a':  #ld b,a: 47
					comp += u8(0x47)
				elif isnum(args[1]):  #ld b,xx: 06xx
					comp += u8(0x06) + u8(args[1])
			elif args[0] == 'c':
				if args[1] == 'b':  #ld c,b: 48
					comp += u8(0x48)
				elif args[1] == 'c':  #ld c,c: 49
					comp += u8(0x49)
				elif args[1] == 'd':  #ld c,d: 4a
					comp += u8(0x4a)
				elif args[1] == 'e':  #ld c,e: 4b
					comp += u8(0x4b)
				elif args[1] == 'h':  #ld c,h: 4c
					comp += u8(0x4c)
				elif args[1] == 'l':  #ld c,l: 4d
					comp += u8(0x4d)
				elif args[1] == '(hl)':  #ld c,(hl): 4e
					comp += u8(0x4e)
				elif args[1] == 'a':  #ld c,a: 4f
					comp += u8(0x4f)
				elif isnum(args[1]):  #ld c,xx: 0exx
					comp += u8(0x0e) + u8(args[1])
			elif args[0] == 'd':
				if args[1] == 'b':  #ld d,b: 50
					comp += u8(0x50)
				elif args[1] == 'c':  #ld d,c: 51
					comp += u8(0x51)
				elif args[1] == 'd':  #ld d,d: 52
					comp += u8(0x52)
				elif args[1] == 'e':  #ld d,e: 53
					comp += u8(0x53)
				elif args[1] == 'h':  #ld d,h: 54
					comp += u8(0x54)
				elif args[1] == 'l':  #ld d,l: 55
					comp += u8(0x55)
				elif args[1] == '(hl)':  #ld d,(hl): 56
					comp += u8(0x56)
				elif args[1] == 'a':  #ld d,a: 57
					comp += u8(0x57)
				elif isnum(args[1]):  #ld d,xx: 16xx
					comp += u8(0x16) + u8(args[1])
			elif args[0] == 'e':
				if args[1] == 'b':  #ld e,b: 58
					comp += u8(0x58)
				elif args[1] == 'c':  #ld e,c: 59
					comp += u8(0x59)
				elif args[1] == 'd':  #ld e,d: 5a
					comp += u8(0x5a)
				elif args[1] == 'e':  #ld e,e: 5b
					comp += u8(0x5b)
				elif args[1] == 'h':  #ld e,h: 5c
					comp += u8(0x5c)
				elif args[1] == 'l':  #ld e,l: 5d
					comp += u8(0x5d)
				elif args[1] == '(hl)':  #ld e,(hl): 5e
					comp += u8(0x5e)
				elif args[1] == 'a':  #ld e,a: 5f
					comp += u8(0x5f)
				elif isnum(args[1]):  #ld e,xx: 1exx
					comp += u8(0x1e) + u8(args[1])
			elif args[0] == 'h':
				if args[1] == 'b':  #ld h,b: 60
					comp += u8(0x60)
				elif args[1] == 'c':  #ld h,c: 61
					comp += u8(0x61)
				elif args[1] == 'd':  #ld h,d: 62
					comp += u8(0x62)
				elif args[1] == 'e':  #ld h,e: 63
					comp += u8(0x63)
				elif args[1] == 'h':  #ld h,h: 64
					comp += u8(0x64)
				elif args[1] == 'l':  #ld h,l: 65
					comp += u8(0x65)
				elif args[1] == '(hl)':  #ld h,(hl): 66
					comp += u8(0x66)
				elif args[1] == 'a':  #ld h,a: 67
					comp += u8(0x67)
				elif isnum(args[1]):  #ld h,xx: 26xx
					comp += u8(0x26) + u8(args[1])
			elif args[0] == 'l':
				if args[1] == 'b':  #ld l,b: 68
					comp += u8(0x68)
				elif args[1] == 'c':  #ld l,c: 69
					comp += u8(0x69)
				elif args[1] == 'd':  #ld l,d: 6a
					comp += u8(0x6a)
				elif args[1] == 'e':  #ld l,e: 6b
					comp += u8(0x6b)
				elif args[1] == 'h':  #ld l,h: 6c
					comp += u8(0x6c)
				elif args[1] == 'l':  #ld l,l: 6d
					comp += u8(0x6d)
				elif args[1] == '(hl)':  #ld l,(hl): 6e
					comp += u8(0x6e)
				elif args[1] == 'a':  #ld l,a: 6f
					comp += u8(0x6f)
				elif isnum(args[1]):  #ld l,xx: 2exx
					comp += u8(0x2e) + u8(args[1])
			elif args[0] == '(hl)':
				if args[1] == 'b':  #ld (hl),b: 70
					comp += u8(0x70)
				elif args[1] == 'c':  #ld (hl),c: 71
					comp += u8(0x71)
				elif args[1] == 'd':  #ld (hl),d: 72
					comp += u8(0x72)
				elif args[1] == 'e':  #ld (hl),e: 73
					comp += u8(0x73)
				elif args[1] == 'h':  #ld (hl),h: 74
					comp += u8(0x74)
				elif args[1] == 'l':  #ld (hl),l: 75
					comp += u8(0x75)
				#no ld (hl),(hl), halt instead
				elif args[1] == 'a':  #ld (hl),a: 77
					comp += u8(0x77)
				elif isnum(args[1]):  #ld (hl),xx: 36xx
					comp += u8(0x36) + u8(args[1])
			elif args[0] == 'a':
				if args[1] == 'b':  #ld a,b: 78
					comp += u8(0x78)
				elif args[1] == 'c':  #ld a,c: 79
					comp += u8(0x79)
				elif args[1] == 'd':  #ld a,d: 7a
					comp += u8(0x7a)
				elif args[1] == 'e':  #ld a,e: 7b
					comp += u8(0x7b)
				elif args[1] == 'h':  #ld a,h: 7c
					comp += u8(0x7c)
				elif args[1] == 'l':  #ld a,l: 7d
					comp += u8(0x7d)
				elif args[1] == '(hl)':  #ld a,(hl): 7e
					comp += u8(0x7e)
				elif args[1] == 'a':  #ld a,a: 7f
					comp += u8(0x7f)
				elif isnum(args[1]):  #ld a,xx: 3exx
					comp += u8(0x3e) + u8(args[1])
				elif args[1] == '(bc)':  #ld a,(bc): 0a
					comp += u8(0x0a)
				elif args[1] == '(de)':  #ld a,(de): 1a
					comp += u8(0x1a)
				elif args[1] == '(hl+)':  #ldi a,(hl): 2a
					comp += u8(0x2a)
				elif args[1] == '(hl-)':  #ldd a,(hl): 3a
					comp += u8(0x3a)
				elif args[1] == '(c)':  #ld a,(c): e200?
					comp += u16(0xe200)
				elif isptr(args[1]):  #ld a,(xxxx): faxxxx
					comp += u8(0xfa) + u16(args[1])
			elif isptr(args[0]):
				if args[0] == '(bc)' and args[1] == 'a':  #ld (bc),a: 02
					comp += u8(0x02)
				elif args[0] == '(de)' and args[1] == 'a':  #ld (de),a: 12
					comp += u8(0x12)
				elif args[0] == '(hl+)' and args[1] == 'a':  #ldi (hl),a: 22
					comp += u8(0x22)
				elif args[0] == '(hl-)' and args[1] == 'a':  #ldd (hl),a: 32
					comp += u8(0x32)
				elif args[0] == '(c)' and args[1] == 'a':  #ld (c),a: f200
					comp += u16(0xf200)
				elif args[1] == 'a':  #ld (xxxx),a: eaxxxx
					comp += u8(0xea) + u16(args[0])
				elif args[1] == 'sp':  #ld sp,xxxx: 08xxcc
					comp += u8(0x08) + u16(args[0])
			elif isreg16(args[0]):
				if args[0] == 'bc' and isnum(args[1]):  #ld bc,xxxx: 01xxxx
					comp += u8(0x01) + u16(args[1])
				elif args[0] == 'de' and isnum(args[1]):  #ld de,xxxx: 11xxxx
					comp += u8(0x11) + u16(args[1])
				elif args[0] == 'hl' and isnum(args[1]):  #ld hl,xxxx: 21xxxx
					comp += u8(0x21) + u16(args[1])
				elif args[0] == 'sp' and isnum(args[1]):  #ld sp,xxxx: 31xxxx
					comp += u8(0x31) + u16(args[1])
				elif args[0] == 'hl' and args[1].startswith('sp+'):  #ld hl,sp+xx: f8xx
					comp += u8(0xf8) + u8(args[1].replace('sp+', ''))
				elif args[0] == 'sp' and args[1] == 'hl':  #ld sp,hl: f9
					comp += u8(0xf9)
		elif op == 'ldh':
			if args[0] == 'a':  #ldh a,(xx): f0xx
				comp += u8(0xf0) + u8(args[1])
			else:  #ldh (xx),a: e0xx
				comp += u8(0xe0) + u8(args[0])
		elif op == 'ldi':
			if args[0] == 'a':  #ldi a,(hl): 2a
				comp += u8(0x2a)
			else:  #ldi (hl),a: 22
				comp += u8(0x22)
		elif op == 'ldd':
			if args[0] == 'a':  #ldd a,(hl): 3a
				comp += u8(0x3a)
			else:  #ldd (hl),a: 33
				comp += u8(0x33)
		elif op == 'inc':
			if args[0] == 'bc':  #inc bc: 03
				comp += u8(0x03)
			elif args[0] == 'de':  #inc de: 13
				comp += u8(0x13)
			elif args[0] == 'hl':  #inc hl: 23
				comp += u8(0x23)
			elif args[0] == 'sp':  #inc sp: 33
				comp += u8(0x33)
			elif args[0] == 'b':  #inc b: 04
				comp += u8(0x04)
			elif args[0] == 'd':  #inc d: 14
				comp += u8(0x14)
			elif args[0] == 'h':  #inc h: 24
				comp += u8(0x24)
			elif args[0] == '(hl)':  #inc (hl): 34
				comp += u8(0x34)
			elif args[0] == 'c':  #inc c: 0c
				comp += u8(0x0c)
			elif args[0] == 'e':  #inc e: 1c
				comp += u8(0x1c)
			elif args[0] == 'l':  #inc l: 2c
				comp += u8(0x2c)
			elif args[0] == 'a':  #inc a: 3c
				comp += u8(0x3c)
		elif op == 'dec':
			if args[0] == 'bc':  #dec bc: 0b
				comp += u8(0x0b)
			elif args[0] == 'de':  #dec de: 1b
				comp += u8(0x1b)
			elif args[0] == 'hl':  #dec hl: 2b
				comp += u8(0x2b)
			elif args[0] == 'sp':  #dec sp: 3b
				comp += u8(0x3b)
			elif args[0] == 'b':  #dec b: 05
				comp += u8(0x05)
			elif args[0] == 'd':  #dec d: 15
				comp += u8(0x15)
			elif args[0] == 'h':  #dec h: 25
				comp += u8(0x25)
			elif args[0] == '(hl)':  #dec (hl): 35
				comp += u8(0x35)
			elif args[0] == 'c':  #dec c: 0d
				comp += u8(0x0d)
			elif args[0] == 'e':  #dec e: 1d
				comp += u8(0x1d)
			elif args[0] == 'l':  #dec l: 2d
				comp += u8(0x2d)
			elif args[0] == 'a':  #dec a: 3d
				comp += u8(0x3d)
		elif op == 'ret':
			if args[0] == 'nz':  #ret nz: c0
				comp += u8(0xc0)
			elif args[0] == 'nc':  #ret nc: d0
				comp += u8(0xd0)
			elif args[0] == 'z':  #ret z: c8
				comp += u8(0xc8)
			elif args[0] == 'c':  #ret c: d8
				comp += u8(0xd8)
			elif args[0] == '':  #ret: c9
				comp += u8(0xc9)
			else:
				error('Opcode %d: %s: Unknown condition %s' % (i, ln, args[0]))
		elif op == 'call':
			if args[1] in labels.keys():
				args[1] = labels[args[1]]
			if args[0] == 'nz':  #call nz,xxxx: c4xxxx
				comp += u8(0xc4) + u16(args[1])
			elif args[0] == 'nc':  #call nc,xxxx: d4xxxx
				comp += u8(0xd4) + u16(args[1])
			elif args[0] == 'z':  #call z,xxxx: ccxxxx
				comp += u8(0xcc) + u16(args[1])
			elif args[0] == 'c':  #call c,xxxx: dcxxxx
				comp += u8(0xdc) + u16(args[1])
			elif args[0] == 'nn':  #call xxxx: cdxxxx
				comp += u8(0xcd) + u16(args[1])
			else:
				error('Opcode %d: %s: Unknown condition %s' % (i, ln, args[0]))
		elif op == 'jp':
			if args[1] in labels.keys():
				args[1] = labels[args[1]]
			if args[0] == 'nz':  #jp nz,xxxx: c2xxxx
				comp += u8(0xc2) + u16(args[1])
			elif args[0] == 'nc':  #jp nc,xxxx: d2xxxx
				comp += u8(0xd2) + u16(args[1])
			elif args[0] == 'z':  #jp z,xxxx: caxxxx
				comp += u8(0xca) + u16(args[1])
			elif args[0] == 'c':  #jp c,xxxx: daxxxx
				comp += u8(0xda) + u16(args[1])
			elif args[0] == 'nn':  #jp xxxx: cdxxxx
				if args[1] == '(hl)':  #jp (hl): e9
					comp += u8(0xe9)
				else:
					comp += u8(0xc3) + u16(args[1])
			else:
				error('Opcode %d: %s: Unknown condition %s' % (i, ln, args[0]))
		elif op == 'add':
			if args[0] == 'a' and not isnum(args[1]):  #add a,x: 8x
				comp += u8(0x80 + REG_INDICS[args[1]])
			elif args[0] == 'a' and isnum(args[1]):  #add a,xx: c6xx
				comp += u8(0xc6) + u8(args[1])
			elif args[1] == 'bc':  #add hl,bc: 09
				comp += u8(0x09)
			elif args[1] == 'de':  #add hl,de: 19
				comp += u8(0x19)
			elif args[1] == 'hl':  #add hl,hl: 29
				comp += u8(0x29)
			elif args[1] == 'sp':  #add hl,sp: 39
				comp += u8(0x39)
			elif args[0] == 'sp':  #add sp,xx: e8xx
				comp += u8(0xe8) + u8(args[1])
		elif op == 'sub':
			if args[0] == 'a' and not isnum(args[1]):  #sub a,x: 9x
				comp += u8(0x90 + REG_INDICS[args[1]])
			elif args[0] == 'a' and isnum(args[1]):  #sub a,xx: d6xx
				comp += u8(0xd6) + u8(args[1])
		elif op == 'and':
			if args[0] == 'a' and not isnum(args[1]):  #and a,x: ax
				comp += u8(0xa0 + REG_INDICS[args[1]])
			elif args[0] == 'a' and isnum(args[1]):  #and a,xx: e6xx
				comp += u8(0xe6) + u8(args[1])
		elif op == 'or':
			if args[0] == 'a' and not isnum(args[1]):  #or a,x: bx
				comp += u8(0xb0 + REG_INDICS[args[1]])
			elif args[0] == 'a' and isnum(args[1]):  #or a,xx: f6xx
				comp += u8(0xf6) + u8(args[1])
		elif op == 'adc':
			if args[0] == 'a' and not isnum(args[1]):  #adc a,x: 8x
				comp += u8(0x88 + REG_INDICS[args[1]])
			elif args[0] == 'a' and isnum(args[1]):  #adc a,xx: cexx
				comp += u8(0xce) + u8(args[1])
		elif op == 'sbc':
			if args[0] == 'a' and not isnum(args[1]):  #sbc a,x: 9x
				comp += u8(0x98 + REG_INDICS[args[1]])
			elif args[0] == 'a' and isnum(args[1]):  #sbc a,xx: dexx
				comp += u8(0xde) + u8(args[1])
		elif op == 'xor':
			if args[0] == 'a' and not isnum(args[1]):  #xor a,x: ax
				comp += u8(0xa8 + REG_INDICS[args[1]])
			elif args[0] == 'a' and isnum(args[1]):  #xor a,xx: eexx
				comp += u8(0xee) + u8(args[1])
		elif op == 'cp':
			if args[0] == 'a' and not isnum(args[1]):  #cp a,x: bx
				comp += u8(0xb8 + REG_INDICS[args[1]])
			elif args[0] == 'a' and isnum(args[1]):  #cp a,xx: fexx
				comp += u8(0xfe) + u8(args[1])
		elif op == 'pop':
			if args[0] == 'bc':  #pop bc: c1
				comp += u8(0xc1)
			elif args[0] == 'de':  #pop de: d1
				comp += u8(0xd1)
			elif args[0] == 'hl':  #pop hl: e1
				comp += u8(0xe1)
			elif args[0] == 'af':  #pop af: f1
				comp += u8(0xf1)
		elif op == 'push':
			if args[0] == 'bc':  #push bc: c5
				comp += u8(0xc5)
			elif args[0] == 'de':  #push de: d5
				comp += u8(0xd5)
			elif args[0] == 'hl':  #push hl: e5
				comp += u8(0xe5)
			elif args[0] == 'af':  #push af: f5
				comp += u8(0xf5)
		elif op == 'rlc':  #rlc x: cb0x
			comp += u8(0xcb) + u8(REG_INDICS[args[0]])
		elif op == 'rrc':  #rrc x: cb0x
			comp += u8(0xcb) + u8(REG_INDICS[args[0]] + 0x08)
		elif op == 'rl':  #rl x: cb1x
			comp += u8(0xcb) + u8(REG_INDICS[args[0]] + 0x10)
		elif op == 'rr':  #rr x: cf1x
			comp += u8(0xcb) + u8(REG_INDICS[args[0]] + 0x18)
		elif op == 'sla':  #sla x: cb2x
			comp += u8(0xcb) + u8(REG_INDICS[args[0]] + 0x20)
		elif op == 'sra':  #sra x: cb2x
			comp += u8(0xcb) + u8(REG_INDICS[args[0]] + 0x28)
		elif op == 'swap':  #swap x: cb3x
			comp += u8(0xcb) + u8(REG_INDICS[args[0]] + 0x30)
		elif op == 'srl':  #srl x: cb3x
			comp += u8(0xcb) + u8(REG_INDICS[args[0]] + 0x38)
		elif op == 'bit':  #bit n,x: cbnx
			comp += u8(0xcb) + u8((8 * toint(args[0])) + REG_INDICS[args[1]] + 0x40)
		elif op == 'res':  #res n,x: cbnx
			comp += u8(0xcb) + u8((8 * toint(args[0])) + REG_INDICS[args[1]] + 0x80)
		elif op == 'set':  #set n,x: cbnx
			comp += u8(0xcb) + u8((8 * toint(args[0])) + REG_INDICS[args[1]] + 0xc0)
		else:
			print('Unrecognized opcode %s' % op)
	return comp


def assemble(sections, labels):
	assembled = {}
	for offset in sections.keys():
		assembled[offset] = _assemble(sections[offset], offset, labels)
	return assembled


def getsize(code):
	size = 0
	for ln in code:
		if ln.endswith(':'):
			continue
		l = lsplit(ln)
		op = l[0]
		if len(l) == 1:
			args = []
		else:
			args = [el.strip() for el in l[1].split(',')]
		if ln.startswith('.incbin'):
			f = bread(l[1].strip('\'"'))
			size += len(f)
		else:
			if op in OPS_1B:
				size += 1
			elif op in OPS_2B:
				size += 2
			elif op in OPS_3B:
				size += 3
			else:
				if op == 'ld':
					if isreg(args[0]) and isreg(args[1]):
						size += 1
					elif isreg16(args[0]) and isnum(args[1]):
						size += 3
					elif isreg8(args[0]) and isnum(args[1]):
						size += 2
					elif isptr(args[0]) and args[1] == 'sp':
						size += 3
					elif args[0] in ('a', '(c)') and args[1] in ('a', '(c)'):
						size += 2
					elif args[0] == 'hl' and args[1].startswith('sp+'):
						size += 2
					elif (isptr(args[0]) and args[1] == 'a') or (args[0] == 'a' and isptr(args[1])):
						size += 3
				elif op == 'jp':
					if args[0] == '(hl)':
						size += 1
					else:
						size += 3
				elif op in NUM_OPS:
					if isnum(args[1]):
						size += 2
					else:
						size += 1
				else:
					paf
					
	return size
