# -*- coding:utf-8 -*-
from util import error
from util.utils import ClsFunc
from util import rawutil

class decompressLZH8 (ClsFunc, rawutil.TypeReader):
	def main(self, input, out, verbose):
		self.byteorder = '<'
		self.verbose = verbose
		self.input = input
		self.read_header(input)
		self.decompress(input, out)
	
	def read_header(self, input):
		self.input.seek(0)
		magic = ord(input.read(1))
		if magic != 0x40:
			error.InvalidMagicError('Invalid magic 0x%02x, expected 0x40' % magic)
		self.decsize = self.unpack_from('U', input, 1)[0]
		if self.decsize == 0:
			self.decsize = self.unpack_from('I', input, 4)[0]
		if self.decsize == 0:  #still
			raise RuntimeError('INTERNAL. SHOULD BE CAUGHT (Recognition error)')
		#File pointer is now at the right place to start decompressing
	
	def getbits(self, count):
		final = 0
		for i in range(count):
			self.bitptr -= 1
			self.masterptr += 1
			if self.bitptr < 0:
				self.bitptr = 7
				val = self.input.read(1)
				if val == b'':
					return None
				self.current = ord(val)
			bit = (self.current >> self.bitptr) & 1
			final |= bit << ((count - 1) - i)
		return final
	
	def decompress(self, input, out):
		self.bitptr = 8
		self.masterptr = 16
		length = ord(self.input.read(1)) + (ord(self.input.read(1)) << 8)
		self.current = ord(self.input.read(1))
		lentbl_datalen = (length + 1) * 4
		lentbl_tablelen = (1 << 9) * 2
		lentbl = [0] * lentbl_tablelen
		i = 1
		while self.masterptr < (lentbl_datalen * 8) - 8:
			if i >= lentbl_tablelen:
				break
			lentbl[i] = self.getbits(9)
			i += 1
		while self.masterptr < lentbl_datalen * 8:
			self.getbits(1)
		
		startptr = self.masterptr
		disptbl_datalen = (self.getbits(8) + 1) * 4
		disptbl_tablelen = (1 << 5) * 2
		disptbl = [0] * disptbl_tablelen
		i = 1
		while self.masterptr < startptr + (disptbl_datalen * 8):
			if i >= disptbl_tablelen:
				break
			disptbl[i] = self.getbits(5)
			i += 1
		while self.masterptr < startptr + (disptbl_datalen * 8):
			self.getbits(1)
		if self.masterptr > startptr + (disptbl_datalen * 8):  #ugly
			diff = self.masterptr - (startptr + (disptbl_datalen * 8))
			self.masterptr -= diff
			self.bitptr += diff
			while self.bitptr > 7:
				self.bitptr -= 8
				self.input.seek(-1, 1)
		
		outsize = 0
		out.write(b'\x00' * self.decsize)
		out.seek(0)
		while outsize < self.decsize:
			lentbl_offset = 1
			while True:
				next_lenchild = self.getbits(1)
				lennode_payload = lentbl[lentbl_offset] & 0x7f
				next_lentbl_offset = (lentbl_offset // 2 * 2) + (lennode_payload + 1) * 2 + bool(next_lenchild)
				next_lenchild_isleaf = lentbl[lentbl_offset] & (0x100 >> next_lenchild)
				if next_lenchild_isleaf:
					length = lentbl[next_lentbl_offset]
					if length < 0x100:
						out.write(bytes((length, )))
						outsize += 1
					else:
						length = (length & 0xff) + 3
						disptbl_offset = 1
						while True:
							next_dispchild = self.getbits(1)
							dispnode_payload = disptbl[disptbl_offset] & 7
							next_disptbl_offset = (disptbl_offset // 2 * 2) + (dispnode_payload + 1) * 2 + bool(next_dispchild)
							next_dispchild_isleaf = disptbl[disptbl_offset] & (0x10 >> next_dispchild)
							if next_dispchild_isleaf:
								dispbs = disptbl[next_disptbl_offset]
								disp = 0
								if dispbs != 0:
									disp = 1
									for i in range(0, dispbs - 1):
										disp <<= 1
										disp |= self.getbits(1)
								disp += 1
								if length > disp:
									out.seek(-disp, 1)
									buf = bytearray(out.read(disp + length))
									#buf[disp: disp + count] = buf[0: count]
									for j in range(length):
										buf[disp + j] = buf[j]
									out.seek(-(disp + length), 1)
									out.write(buf)
								else:
									out.seek(-disp, 1)
									ref = out.read(length)
									out.seek((disp - length), 1)
									out.write(ref)
								outsize = out.tell()
								if outsize >= self.decsize:
									break
								break
							else:
								assert next_disptbl_offset != disptbl_offset
								disptbl_offset = next_disptbl_offset
					break
				else:
					assert next_lentbl_offset != lentbl_offset
					lentbl_offset = next_lentbl_offset
