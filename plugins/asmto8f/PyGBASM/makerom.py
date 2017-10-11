# -*- coding:utf-8 -*-

import binascii
from ._intern import *

NINTENDO_LOGO = binascii.unhexlify('CEED6666CC0D000B03730083000C000D0008111F8889000EDCCC6EE6DDDDD999BBBB67636E0EECCCDDDC999FBBB9333E')
ROM_SIZES = {
	0x8000: 0x00,
	0x10000: 0x01,
	0x20000: 0x02,
	0x40000: 0x03,
	0x80000: 0x04,
	0x100000: 0x05,
	0x200000: 0x06,
	0x120000: 0x52,
	0x140000: 0x53,
	0x180000: 0x54
}
RAM_SIZES = {
	0: 0x00,
	0x800: 0x01,
	0x2000: 0x02,
	0x8000: 0x03,
	0x20000: 0x04
}

def makerom(code, info, outname):
	rom = open(outname, 'wb+')
	romsize = toint(info['romsize'][0])
	ramsize = toint(info['ramsize'][0])
	if 'emptyfill' in info.keys():
		rom.write(bytes(romsize * [toint(info['emptyfill'][0])]))
	else:
		rom.write(bytes([0] * romsize))
	rom.seek(0x104)
	rom.write(NINTENDO_LOGO)
	rom.seek(0x134)
	rom.write(info['name'][0].strip('"\'').upper().encode('ascii'))
	rom.seek(0x143)
	rom.write(u8(0x80) if 'romcgb' in info.keys() else u8(0))
	rom.write(info['licenseecodenew'][0].strip('"\'').encode('ascii'))
	rom.write(u8(0))
	rom.write(u8(ROM_SIZES[romsize]))
	rom.write(u8(RAM_SIZES[ramsize]))
	rom.write(u8(info['location'][0]))
	rom.write(u16(0x3300))
	rom.seek(0x134)
	header = rom.read(25)
	hdrcrc = 0
	for b in header:
		hdrcrc = hdrcrc - b - 1
	rom.seek(0x14d)
	rom.write(u16(hdrcrc % 256))
	for offset in code.keys():
		sec = code[offset]
		rom.seek(offset)
		rom.write(sec)
	rom.seek(0)
	romcnt = rom.read()
	romsum = sum(romcnt) & 0xffff
	rom.seek(0x14e)
	rom.write(u16b(romsum))
	rom.close()

def disrom(romname):
	pass
