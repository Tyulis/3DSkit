# -*- coding:utf-8 -*-
import array
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
	def main(self, content, verbose):
		self.byteorder = '>'
		self.verbose = verbose
		self.readhdr(content)
		return self.decompress()
	
	def readhdr(self, content):
		if content[0] != 0x11:
			error('Invalid magic 0x%02x, expected 0x11' % content[0], 301)
		self.decsize = self.unpack_from('<U', content[:4], 1)[0]
		self.data = content[4:]
	
	def decompress(self):
		ptr = 0
		final = []
		while len(final) < self.decsize:
			flags = self.data[ptr]
			ptr += 1
			mask = 1 << 7
			for i in range(8):
				#Todo: Make this a bit less horrible
				if flags & mask == 0:
					byte = self.data[ptr]
					ptr += 1
					final.append(byte)
				else:
					byte1 = self.data[ptr]
					ptr += 1
					indic = byte1 >> 4
					if indic == 0:
						byte2, byte3 = self.data[ptr: ptr + 2]
						ptr += 2
						count = (byte1 << 4) + (byte2 >> 4) + 0x11
						disp = ((byte2 & 0xf) << 8) + byte3 + 1
					elif indic == 1:
						byte2, byte3, byte4 = self.data[ptr: ptr + 3]
						ptr += 3
						count = ((byte1 & 0xf) << 12) + (byte2 << 4) + (byte3 >> 4) + 0x111
						disp = ((byte3 & 0xf) << 8) + byte4 + 1
					else:
						byte2, ptr = self.uint8(self.data, ptr)
						count = indic + 1
						disp = ((byte1 & 0xf) << 8) + byte2 + 1
					for i in range(0, count):
						final.append(final[-disp])
				l = len(final)
				if len(final) >= self.decsize:
					break
				mask >>= 1
			if len(final) >= self.decsize:
					break
		return bytes(final)
