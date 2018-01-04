# -*- coding:utf-8 -*-
from util import error
from util.utils import ClsFunc
import util.rawutil as rawutil
from collections import defaultdict
from operator import itemgetter


#TODO: Change to the new algorithm like LZ11
class compressLZ10 (ClsFunc, rawutil.TypeWriter):
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
		self.out.write(b'\x10')
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
						if size >= 0xf + 3:
							break
						prevbyte = data[subptr]
						newbyte = data[ptr]
				disp -= 1
				count = size - 3
				byte1 = (count << 4) + (disp >> 8)
				byte2 = disp & 0xff
				buf[bufpos: bufpos + 2] = (byte1, byte2)
				bufpos += 2
				buf[0] |= 1 << (7 - bufblk)
			bufblk += 1
		self.out.write(buf[:bufpos])
		ptr = self.out.tell()
		padding = 4 - (ptr % 4 or 4)
		self.out.write(b'\x00' * padding)


class decompressLZ10 (ClsFunc, rawutil.TypeReader):
	def main(self, input, out, verbose):
		self.byteorder = '>'
		self.input = input
		self.out = out
		self.verbose = verbose
		self.readhdr(input)
		return self.decompress()
	
	def readhdr(self, content):
		magic = ord(content.read(1))
		if magic != 0x10:
			error.InvalidMagicError('Invalid magic 0x%02x, expected 0x10' % magic)
		self.decsize = self.unpack_from('<U', content, 1)[0]
		if self.decsize == 0:
			raise RuntimeError('INTERNAL. SHOULD BE CAUGHT (Recognition error)')
	
	def decompress(self):
		self.input.seek(4)
		self.out.seek(0)
		'''
		while len(final) < self.decsize:
			flags = self.tobits(self.data[ptr])
			ptr += 1
			for flag in flags:
				if flag == 0:
					byte, ptr = self.uint8(self.data, ptr)
					final.append(byte)
				else:
					infobs = self.unpack_from('H', self.data, ptr)[0]
					ptr += 2
					count = (infobs >> 12) + 3
					disp = (infobs & 0xfff) + 1
					for i in range(0, count):
						final.append(final[-disp])
				if len(final) >= self.decsize:
					break
		ret = bytes(final)
		return ret
		'''
		outlen = 0
		self.out.write(b'\x00' * self.decsize)
		self.out.seek(0)
		block = -1
		while outlen < self.decsize:
			if block == -1:
				flags = ord(self.input.read(1))
				block = 7
			mask = 1 << block
			if flags & mask == 0:
				byte = self.input.read(1)
				self.out.write(byte)
			else:
				byte1 = ord(self.input.read(1))
				byte2 = ord(self.input.read(1))
				count = (byte1 >> 4) + 3
				disp = ((byte1 & 0xf) << 8) + byte2 + 1
				if count > disp:
					self.out.seek(-disp, 1)
					buf = bytearray(self.out.read(disp + count))
					#buf[disp: disp + count] = buf[0: count]
					for j in range(count):
						buf[disp + j] = buf[j]
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
