# -*- coding:utf-8 -*-

import numpy as np


def compressLZ11(input, output, insize):
	ptr = 0
	buf = np.zeros(33, dtype=np.uint8)
	bufpos = 1
	bufblk = 0
	outpos = 0
	while ptr < insize:
		if bufblk == 8:
			output[outpos: outpos + bufpos] = buf[:bufpos]
			outpos += bufpos
			buf[0] = 0
			bufblk = 0
			bufpos = 1
		min = ptr - 4096
		if min < 0:
			min = 0
		sub = input[ptr: ptr + 3].tostring()
		if len(sub) < 3:
			buf[bufpos] = input[ptr]
			bufpos += 1
			ptr += 1
			bufblk += 1
			continue
		idx = input[min: ptr].tostring().find(sub)
		if idx == -1:
			buf[bufpos] = input[ptr]
			ptr += 1
			bufpos += 1
		else:
			pos = idx + min
			disp = ptr - pos
			size = 3
			subptr = pos + 3
			ptr += 3
			if ptr < insize:
				prevbyte = input[subptr]
				newbyte = input[ptr]
				while prevbyte == newbyte and subptr < insize - 1 and ptr < insize - 1:
					subptr += 1
					ptr += 1
					size += 1
					if size >= 0x10110:
						break
					prevbyte = input[subptr]
					newbyte = input[ptr]
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
	output[outpos: outpos + bufpos] = buf[:bufpos]
	outpos += 4 - (outpos % 4 or 4)
	return outpos

def decompressLZ11(input, output, insize, outsize):
	outlen = 0
	inpos = 0
	block = -1
	while outlen < outsize:
		if block == -1:
			flags = input[inpos]
			inpos += 1
			block = 7
		mask = 1 << block
		if not flags & mask:
			byte = input[inpos]
			inpos += 1
			output[outlen] = byte
			outlen += 1
		else:
			byte1 = input[inpos]
			inpos += 1
			indic = byte1 >> 4
			if indic == 0:
				byte2, byte3 = input[inpos], input[inpos + 1]
				inpos += 2
				count = (byte1 << 4) + (byte2 >> 4) + 0x11
				disp = ((byte2 & 0xf) << 8) + byte3 + 1
			elif indic == 1:
				byte2, byte3, byte4 = input[inpos], input[inpos + 1], input[inpos + 2]
				inpos += 3
				count = ((byte1 & 0xf) << 12) + (byte2 << 4) + (byte3 >> 4) + 0x111
				disp = ((byte3 & 0xf) << 8) + byte4 + 1
			else:
				byte2 = input[inpos]
				inpos += 1
				count = indic + 1
				disp = ((byte1 & 0xf) << 8) + byte2 + 1
			refpos = outlen - disp
			for j in range(count):
				output[outlen + j] = output[refpos + j]
			outlen += count
		if outlen >= outsize:
			break
		block -= 1
