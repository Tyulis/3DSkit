# -*- coding:utf-8 -*-
from ._intern import *

OPS_1B = {
	0x00: 'nop',
	0x02: 'ld (bc), a',
	0x03: 'inc bc',
	0x04: 'inc b',
	0x05: 'dec b',
	0x07: 'rlca',
	0x09: 'add hl, bc',
	0x0a: 'ld a, (bc)',
	0x0b: 'dec bc',
	0x0c: 'inc c',
	0x0d: 'dec c',
	0x0f: 'rrca',
	0x10: 'stop',
	0x12: 'ld (de), a',
	0x13: 'inc de',
	0x14: 'inc d',
	0x15: 'dec d',
	0x17: 'rla',
	0x19: 'add hl, de',
	0x1a: 'ld a, (de)',
	0x1b: 'dec de',
	0x1c: 'inc e',
	0x1d: 'dec e',
	0x1f: 'rra',
	0x22: 'ldi (hl), a',
	0x23: 'inc hl',
	0x24: 'inc h',
	0x25: 'dec h',
	0x27: 'daa',
	0x29: 'add hl, hl',
	0x2a: 'ldi a, (hl)',
	0x2b: 'dec hl',
	0x2c: 'inc l',
	0x2d: 'dec l',
	0x2f: 'cpl',
	0x32: 'ldd (hl), a',
	0x33: 'inc sp',
	0x34: 'inc (hl)',
	0x35: 'dec (hl)',
	0x37: 'scf',
	0x39: 'add hl, sp',
	0x3a: 'ldd a, (hl)',
	0x3b: 'dec sp',
	0x3c: 'inc a',
	0x3d: 'dec a',
	0x3f: 'ccf',
	0x76: 'halt',
	0xc0: 'ret nz',
	0xc1: 'pop bc',
	0xc5: 'push bc',
	0xc7: 'rst $00',
	0xc8: 'ret z',
	0xc9: 'ret',
	0xcf: 'rst $08',
	0xd0: 'ret nc',
	0xd1: 'pop de',
	0xd5: 'push de',
	0xd7: 'rst $10',
	0xd8: 'ret c',
	0xd9: 'reti',
	0xdf: 'rst $18',
	0xe1: 'pop hl',
	0xe2: 'ld (c), a',
	0xe5: 'push hl',
	0xe7: 'rst $20',
	0xe9: 'jp (hl)',
	0xef: 'rst $28',
	0xf1: 'pop af',
	0xf2: 'ld a, (c)',
	0xf3: 'di',
	0xf5: 'push af',
	0xf7: 'rst $30',
	0xf9: 'ld sp, hl',
	0xfb: 'ei',
	0xff: 'rst $38',
}

regs = ('b', 'c', 'd', 'e', 'h', 'l', '(hl)', 'a')

MATH_OPS = ('add', 'adc', 'sub', 'sbc', 'and', 'xor', 'or', 'cp')


def _disassemble(bin):
	code = []
	f = FakeFile(bin)
	while True:
		try:
			op = f.uint8()
			if op in OPS_1B.keys():
				code.append(OPS_1B[op])
				if op in (0x10, 0xe2, 0xf2):
					nop = f.uint8()
			elif op in range(0x40, 0x80):
				reg1 = (op & 0b00111000) >> 3
				reg2 = op & 0b00000111
				code.append('ld %s, %s' % (regs[reg1], regs[reg2]))
			elif op in range(0x80, 0xc0):
				operation = (op & 0b00111000) >> 3
				reg = op & 0b00000111
				code.append('%s a, %s' % (MATH_OPS[operation], regs[reg]))
			elif op == 0x20:
				code.append('jr nz, $%02x' % f.int8())
			elif op == 0x30:
				code.append('jr nc, $%02x' % f.int8())
			elif op == 0x01:
				code.append('ld bc, $%04x' % f.uint16())
			elif op == 0x11:
				code.append('ld de, $%04x' % f.uint16())
			elif op == 0x21:
				code.append('ld hl, $%04x' % f.uint16())
			elif op == 0x31:
				code.append('ld sp, $%04x' % f.uint16())
			elif op == 0x06:
				code.append('ld b, $%02x' % f.uint8())
			elif op == 0x16:
				code.append('ld d, $%02x' % f.uint8())
			elif op == 0x26:
				code.append('ld h, $%02x' % f.uint8())
			elif op == 0x36:
				code.append('ld (hl), $%02x' % f.uint8())
			elif op == 0x08:
				code.append('ld ($%04x), sp' % f.uint16())
			elif op == 0x18:
				code.append('jr $%02x' % f.int8())
			elif op == 0x28:
				code.append('jr z, $%02x' % f.int8())
			elif op == 0x38:
				code.append('jr c, $%02x' % f.int8())
			elif op == 0x0e:
				code.append('ld c, $%02x' % f.uint8())
			elif op == 0x1e:
				code.append('ld e, $%02x' % f.uint8())
			elif op == 0x2e:
				code.append('ld l, $%02x' % f.uint8())
			elif op == 0x3e:
				code.append('ld a, $%02x' % f.uint8())
			elif op == 0xe0:
				code.append('ldh ($%02x), a' % f.uint8())
			elif op == 0xf0:
				code.append('ldh a, ($%02x)' % f.uint8())
			elif op == 0xc2:
				code.append('jp nz, $%04x' % f.uint16())
			elif op == 0xd2:
				code.append('jp nc, $%04x' % f.uint16())
			elif op == 0xc3:
				code.append('jp $%04x' % f.uint16())
			elif op == 0xc4:
				code.append('call nz, $%04x' % f.uint16())
			elif op == 0xd4:
				code.append('call nc, $%04x' % f.uint16())
			elif op == 0xc6:
				code.append('add a, $%02x' % f.uint8())
			elif op == 0xd6:
				code.append('sub a, $%02x' % f.uint8())
			elif op == 0xe6:
				code.append('and a, $%02x' % f.uint8())
			elif op == 0xf6:
				code.append('or a, $%02x' % f.uint8())
			elif op == 0xe8:
				code.append('add sp, $%02x' % f.uint8())
			elif op == 0xf8:
				code.append('ld hl, sp+$%02x' % f.uint8())
			elif op == 0xca:
				code.append('jp z, $%04x' % f.uint16())
			elif op == 0xda:
				code.append('jp c, $%04x' % f.uint16())
			elif op == 0xea:
				code.append('ld ($%04x), a' % f.uint16())
			elif op == 0xfa:
				code.append('ld a, ($%04x)' % f.uint16())
			elif op == 0xcc:
				code.append('call z, $%04x' % f.uint16())
			elif op == 0xdc:
				code.append('call c, $%04x' % f.uint16())
			elif op == 0xcd:
				code.append('call $%04x' % f.uint16())
			elif op == 0xce:
				code.append('adc a, $%02x' % f.uint8())
			elif op == 0xde:
				code.append('sbc a, $%02x' % f.uint8())
			elif op == 0xee:
				code.append('xor a, $%02x' % f.uint8())
			elif op == 0xfe:
				code.append('cp a, $%02x' % f.uint8())
			else:
				print('Unknown opcode %02x' % op)
		except IOError:
			break
	return '\n'.join(code)

def disassemble(assembled):
	code = ''
	for offset in assembled.keys():
		code += '.org $%06x\n' % offset
		code += _disassemble(assembled[offset])
		code += '\n\n'
	return code
