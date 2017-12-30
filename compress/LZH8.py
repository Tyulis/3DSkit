# -*- coding:utf-8 -*-
from util import error
from util.utils import ClsFunc
import util.rawutil as rawutil


class decompressLZH8 (ClsFunc, rawutil.TypeReader):
	def main(self, content, verbose):
		self.byteorder = '<'
		self.verbose = verbose
		data = self.readhdr(content)
		return self.decompress(data)
	
	def readhdr(self, data):
		if data[0] != 0x40:
			error.InvalidMagicError('Invalid magic 0x%02x, expected 0x40' % data[0])
		self.unco_len = self.unpack_from('U', data, 1)[0]
		hdrend = 4
		if self.unco_len == 0:
			self.unco_len = self.unpack_from('I', data, 4)[0]
			hdrend = 8
		if self.unco_len == 0:
			raise RuntimeError('INTERNAL. SHOULD BE CAUGHT (Recognition error)')
		return data[hdrend:]
	
	def getbits(self, bitnum):
		bits = self.bits[self.bitptr: self.bitptr + bitnum]
		self.bitptr += bitnum
		return int(bits, 2)
	
	def decompress(self, data):
		self.bits = rawutil.bin(data)
		self.bitptr = 16
		lentbl_datalen = (self.uint16(data, 0)[0] + 1) * 4
		lentbl_tablelen = (1 << 9) * 2
		lentbl = [0] * lentbl_tablelen
		i = 1
		while self.bitptr < (lentbl_datalen * 8) - 8:
			if i >= lentbl_tablelen:
				break
			lentbl[i] = self.getbits(9)
			i += 1
		self.bitptr = lentbl_datalen * 8
		
		startptr = self.bitptr
		disptbl_datalen = (self.getbits(8) + 1) * 4
		disptbl_tablelen = (1 << 5) * 2
		disptbl = [0] * disptbl_tablelen
		i = 1
		while self.bitptr < startptr + (disptbl_datalen * 8):
			if i >= disptbl_tablelen:
				break
			disptbl[i] = self.getbits(5)
			i += 1
		self.bitptr = startptr + (disptbl_datalen * 8)
		
		final = []
		j = 0
		while len(final) < self.unco_len:
			j += 1
			lentbl_offset = 1
			while True:
				next_lenchild = self.getbits(1)
				lennode_payload = lentbl[lentbl_offset] & 0x7f
				next_lentbl_offset = (lentbl_offset // 2 * 2) + (lennode_payload + 1) * 2 + bool(next_lenchild)
				next_lenchild_isleaf = lentbl[lentbl_offset] & (0x100 >> next_lenchild)
				if next_lenchild_isleaf:
					length = lentbl[next_lentbl_offset]
					if length < 0x100:
						final.append(length)
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
								for j in range(0, length):
									final.append(final[len(final) - disp - 1])
									if len(final) >= self.unco_len:
										break
								break
							else:
								assert next_disptbl_offset != disptbl_offset
								disptbl_offset = next_disptbl_offset
					break
				else:
					assert next_lentbl_offset != lentbl_offset
					lentbl_offset = next_lentbl_offset
		return bytes(final)
