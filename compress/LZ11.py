# -*- coding:utf-8 -*-
from util.funcops import ClsFunc
import util.rawutil as rawutil


class compressLZ11 (ClsFunc, rawutil.TypeWriter):
	def main(self, file, out, verbose):
		self.byteorder = '>'
		self.verbose = verbose
		self.file = file
		self.out = out
		self.file.seek(0, 2)
		self.datalen = self.file.tell()
		self.file.seek(0)
		self.out.seek(0)
		self.makeheader()
		self.compress()
	
	def makeheader(self):
		self.out.write(b'\x11')
		self.pack('<U', self.datalen, self.out)
	
	def compress(self):
		data = self.file.read()
		ptr = 0
		buf = bytearray(33)
		bufpos = 1
		bufblk = 0
		while ptr < self.datalen:
			if bufblk == 8:
				self.out.write(buf[:bufpos])
				buf[0] = 0
				bufblk = 0
				bufpos = 1
			min = ptr - 4096
			if min < 0:
				min = 0
			sub = data[ptr: ptr + 3]
			if len(sub) < 3:
				buf[bufpos] = data[ptr]
				bufpos += 1
				ptr += 1
				bufblk += 1
				continue
			idx = data[min: ptr].find(sub)
			if idx == -1:
				buf[bufpos] = data[ptr]
				ptr += 1
				bufpos += 1
			else:
				pos = idx + min
				disp = ptr - pos
				size = 3
				subptr = pos + 3
				ptr += 3
				if ptr < self.datalen:
					prevbyte = data[subptr]
					newbyte = data[ptr]
					while prevbyte == newbyte and subptr < self.datalen - 1 and ptr < self.datalen - 1:
						subptr += 1
						ptr += 1
						size += 1
						if size >= 0x10110:
							break
						prevbyte = data[subptr]
						newbyte = data[ptr]
				disp -= 1
				if size > 0xff + 0x11:
					#sh = (1 << 28) | ((size - 0x111) << 12) | (disp - 1)
					#buf[bufpos: bufpos + 4] = self.uint32(sh)
					count = size - 0x111
					byte1 = 0x10 + (count >> 12)
					byte2 = (count >> 4) & 0xff
					byte3 = ((count & 0x0f) << 4) | (disp >> 8)
					byte4 = disp & 0xff
					buf[bufpos: bufpos + 4] = (byte1, byte2, byte3, byte4)
					bufpos += 4
				elif size > 0xf + 0x1:
					#sh = ((size - 0x11) << 12) | (disp - 1)
					#buf[bufpos: bufpos + 3] = self.uint24(sh)
					count = size - 0x11
					byte1 = count >> 4
					byte2 = ((count & 0x0f) << 4) | (disp >> 8)
					byte3 = disp & 0xff
					buf[bufpos: bufpos + 3] = (byte1, byte2, byte3)
					bufpos += 3
				else:
					#sh = ((size - 1) << 12) | (disp - 1)
					#buf[bufpos: bufpos + 2] = self.uint16(sh)
					byte1 = ((size - 1) << 4) | (disp >> 8)
					byte2 = disp & 0xff
					buf[bufpos: bufpos + 2] = (byte1, byte2)
					bufpos += 2
				buf[0] |= 1 << (7 - bufblk)
			bufblk += 1
		self.out.write(buf[:bufpos])
		ptr = self.out.tell()
		padding = 4 - (ptr % 4 or 4)
		self.out.write(b'\x00' * padding)
		

#inspirated from nlzss

class decompressLZ11 (ClsFunc, rawutil.TypeReader):
	def main(self, file, out, verbose):
		self.byteorder = '>'
		self.verbose = verbose
		self.file = file
		self.out = out
		self.file.seek(0)
		self.out.seek(0)
		self.readhdr()
		self.decompress()
	
	def readhdr(self):
		magic, self.decsize = self.unpack_from('<BU', self.file)
		if magic != 0x11:
			error('Invalid magic 0x%02x, expected 0x11' % magic, 301)
	
	def decompress(self):
		outlen = 0
		self.out.write(b'\x00' * self.decsize)
		self.out.seek(0)
		block = -1
		while outlen < self.decsize:
			if block == -1:
				flags = ord(self.file.read(1))
				block = 7
			mask = 1 << block
			if flags & mask == 0:
				byte = self.file.read(1)
				self.out.write(byte)
			else:
				byte1 = ord(self.file.read(1))
				indic = byte1 >> 4
				if indic == 0:
					byte2, byte3 = self.file.read(2)
					count = (byte1 << 4) + (byte2 >> 4) + 0x11
					disp = ((byte2 & 0xf) << 8) + byte3 + 1
				elif indic == 1:
					byte2, byte3, byte4 = self.file.read(3)
					count = ((byte1 & 0xf) << 12) + (byte2 << 4) + (byte3 >> 4) + 0x111
					disp = ((byte3 & 0xf) << 8) + byte4 + 1
				else:
					byte2 = ord(self.file.read(1))
					count = indic + 1
					disp = ((byte1 & 0xf) << 8) + byte2 + 1
				if count > disp:
					self.out.seek(-disp, 1)
					buf = bytearray(self.out.read(disp + count))
					buf[disp: disp + count] = buf[0: count]
					#for j in range(count):
					#	buf[disp + j] = buf[j]
					self.out.seek(-(disp + count), 1)
					self.out.write(buf)
				else:
					self.out.seek(-disp, 1)
					ref = self.out.read(count)
					self.out.seek((disp - count), 1)
					self.out.write(ref)
			outlen = self.out.tell()
			if outlen >= self.decsize:
				break
			block -= 1
