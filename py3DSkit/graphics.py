# -*- coding:utf-8 -*-
# Python translation of c3DSkit/graphics.c

import numpy as np

RGBA8 = 0
RGB8 = 1
RGBA5551 = 2
RGB565 = 3
RGBA4 = 4
LA8 = 5
RG8 = 6
L8 = 7
A8 = 8
LA4 = 9
L4 = 10
A4 = 10
ETC1 = 11
ETC1A4 = 12
BC1 = 13
BC2 = 14
BC3 = 15
BC4 = 16
BC5 = 17
BC6H = 18
BC7 = 19
RGBA8_SRGB = 20
BC1_SRGB = 21
BC2_SRGB = 22
BC3_SRGB = 23
BC4_SNORM = 24
BC5_SNORM = 25
BC6H_SF16 = 26
BC7_SRGB = 27

TEXTURE_FORMATS = {
	'RGBA8': RGBA8, 'RGB8': RGB8, 'RGBA5551': RGBA5551,
	'RGBA4': RGBA4, 'LA8': LA8, 'RG8': RG8, 'L8': L8,
	'A8': A8, 'LA4': LA4, 'L4': L4, 'A4': A4,
	'ETC1': ETC1, 'ETC1A4': ETC1A4, 'BC4': BC4, 'BC4_SNORM': BC4_SNORM,
	#'BC1': BC1, 'BC2': BC2, 'BC3': BC3, 'BC5': BC5,
	#'BC6H': BC6H, 'BC7': BC7, 'RGBA8_SRGB': RGBA8_SRGB,
	#'BC1_SRGB': BC1_SRGB, 'BC2_SRGB': BC2_SRGB,
	#'BC3_SRGB': BC3_SRGB, 
	#'BC5_SNORM': BC5_SNORM, 'BC6H_SF16': BC6H_SF16,
	#'BC7_SRGB': BC7_SRGB,
}

ETC1_MODIFIERS = (
	(2, 8), (5, 17), (9, 29), (13, 42),
	(18, 60), (24, 80), (33, 106), (47, 183)
)

PIXEL_SIZES = {
	RGBA8: 4, RGB8: 3, RGBA5551: 2,
	RGBA4: 2, LA8: 2, RG8: 2,
	L8: 1, A8: 1, LA4: 1
}

BC4_BYTESPERTEXEL = 8

LOG2 = lambda x: np.log10(x) / np.log10(2)


class Swizzle (object):
	def __init__(self, width, bpp, blockheight):
		if blockheight <= 0:
			blockheight = 16
		self.bhmask = blockheight * 8 - 1
		self.bhshift = self._getShift(blockheight * 8)
		self.bppshift = self._getShift(bpp)
		self.gobstride = int(512 * blockheight * np.ceil(width * bpp / 64.0))
		self.xshift = self._getShift(512 * blockheight)
	
	def _getShift(self, value):
		if value == 0:
			return 0
		shift = 0
		while ((value >> shift) & 1) == 0:
			shift += 1
		return shift;
	
	def getSwizzleOffset(self, x, y):
		x <<= self.bppshift
		pos = (y >> self.bhshift) * self.gobstride
		pos += (x >> 6) << self.xshift
		pos += ((y & self.bhmask) >> 3) << 9
		pos += ((x & 0x3f) >> 5) << 8
		pos += ((y & 0x07) >> 1) << 6
		pos += ((x & 0x1f) >> 4) << 5
		pos += ((y & 0x01) >> 0) << 4
		pos += ((x & 0x0f) >> 0) << 0
		return pos
		
def getTextureFormatId(format):
	if format in TEXTURE_FORMATS:
		return TEXTURE_FORMATS[format]
	else:
		return -1

def getPixelSize(format):
	if format in PIXEL_SIZES:
		return PIXEL_SIZES[format]
	else:
		return -1

def _extractTiledTexture(input, output, width, height, format, littleendian):
	tilesx = int(np.ceil(width / 8))
	tilesy = int(np.ceil(height / 8))
	datawidth = 1 << int(np.ceil(LOG2(width)))
	#int dataheight = 1 << (int)ceil(LOG2((double)height));
	totalx = int(np.ceil(datawidth / 8.0))
	#int totaly = (int)ceil((double)dataheight / 8.0);
	pxsize = getPixelSize(format)
	for ytile in range(tilesy):
		for xtile in range(tilesx):
			for ysub in range(2):
				for xsub in range(2):
					for yblock in range(2):
						for xblock in range(2):
							for ypix in range(2):
								for xpix in range(2):
									ypos = ytile * 8 + ysub * 4 + yblock * 2 + ypix
									xpos = xtile * 8 + xsub * 4 + xblock * 2 + xpix
									if xpos >= width or ypos >= height:
										continue
									outpos = (ypos * width + xpos) * 4
									if format == L4 or format == A4:
										shift = xpix * 4
										inpos = ytile * totalx * 32 + xtile * 32 + ysub * 16 + xsub * 8 + yblock * 4 + xblock * 2 + ypix
										byte = input[inpos]
										if format == L4:
											r = g = b = ((byte >> shift) & 0x0F) * 0x11
											a = 0xFF;
										elif format == A4:
											r = g = b = 0xFF
											a = ((byte >> shift) & 0x0F) * 0x11
									else:
										inpos = (ytile * totalx * 64 + xtile * 64 + ysub * 32 + xsub * 16 + yblock * 8 + xblock * 4 + ypix * 2 + xpix) * pxsize
										if littleendian:
											if format == RGBA8:
												r = input[inpos + 3]
												g = input[inpos + 2]
												b = input[inpos + 1]
												a = input[inpos]
											elif format == RGB8:
												r = input[inpos + 2]
												g = input[inpos + 1]
												b = input[inpos]
												a = 0xFF
											elif format == RGBA5551:
												r = (input[inpos + 1] >> 3) * 8.225806451612
												g = (((input[inpos + 1] & 0x07) << 2) | (input[inpos] >> 6)) * 8.225806451612
												b = ((input[inpos] >> 1) & 0x1F) * 8.225806451612
												a = (input[inpos] & 1) * 0xFF
											elif format == RGB565:
												r = (input[inpos + 1] >> 3) * 8.225806451612
												g = (((input[inpos + 1] & 0x07) << 3) | (input[inpos] >> 5)) * 4.0476190476190
												b = (input[inpos] & 0x1F) * 8.225806451612
												a = 0xFF
											elif format == RGBA4:
												r = (input[inpos + 1] >> 4) * 0x11
												g = (input[inpos + 1] & 0x0F) * 0x11
												b = (input[inpos] >> 4) * 0x11
												a = (input[inpos] & 0x0F) * 0x11
											elif format == LA8:
												r = g = b = input[inpos + 1]
												a = input[inpos]
											elif format == RG8:
												r = input[inpos + 1]
												g = input[inpos]
												b = 0
												a = 0xFF
											elif format == L8:
												r = g = b = input[inpos]
												a = 0xFF
											elif format == A8:
												r = g = b = 0xFF
												a = input[inpos]
											elif format == LA4:
												r = g = b = (input[inpos] >> 4) * 0x11
												a = (input[inpos] & 0x0F) * 0x11
										else:
											if format == RGBA8:
												r = input[inpos]
												g = input[inpos + 1]
												b = input[inpos + 2]
												a = input[inpos + 3]
											elif format == RGB8:
												r = input[inpos]
												g = input[inpos + 1]
												b = input[inpos + 2]
												a = 0xFF
											elif format == RGBA5551:
												r = (input[inpos] >> 3) * 8.225806451612
												g = (((input[inpos] & 0x07) << 2) | (input[inpos + 1] >> 6)) * 8.225806451612
												b = ((input[inpos + 1] >> 1) & 0x1F) * 8.225806451612
												a = (input[inpos + 1] & 1) * 0xFF
											elif format == RGB565:
												r = (input[inpos] >> 3) * 8.225806451612
												g = (((input[inpos] & 0x07) << 3) | (input[inpos + 1] >> 5)) * 4.0476190476190
												b = (input[inpos + 1] & 0x1F) * 8.225806451612
												a = 0xFF
											elif format == RGBA4:
												r = (input[inpos] >> 4) * 0x11
												g = (input[inpos] & 0x0F) * 0x11
												b = (input[inpos + 1] >> 4) * 0x11
												a = (input[inpos + 1] & 0x0F) * 0x11
											elif format == LA8:
												r = g = b = input[inpos]
												a = input[inpos + 1]
											elif format == RG8:
												r = input[inpos]
												g = input[inpos + 1]
												b = 0
												a = 0xFF
											elif format == L8:
												r = g = b = input[inpos]
												a = 0xFF
											elif format == A8:
												r = g = b = 0xFF
												a = input[inpos]
											elif format == LA4:
												r = g = b = (input[inpos] >> 4) * 0x11
												a = (input[inpos] & 0x0F) * 0x11
									output[outpos] = r
									output[outpos + 1] = g
									output[outpos + 2] = b
									output[outpos + 3] = a

#ETC1 decompression is inspirated from ObsidianX's 3dstools BFLIM implementation

def ETC1DiffComplement(val, bits):
	if (val >> (bits - 1)) == 0:
		return val
	return val - (1 << bits)

def _extractETC1Texture(input, output, width, height, format):
	hasalpha = (format == ETC1A4)
	tilew = 1 << int(np.ceil(LOG2(width / 8)))
	tileh = 1 << int(np.ceil(LOG2(height / 8)))
	inpos = 0
	alphas = 0xFFFFffffFFFFffff
	color1 = np.zeros(3, dtype=np.uint8)
	color2 = np.zeros(3, dtype=np.uint8)
	for ytile in range(tileh):
		for xtile in range(tilew):
			for yblock in range(2):
				for xblock in range(2):
					if hasalpha:
						alphas = 0
						for i in range(8):
							alphas |= int(input[inpos]) << (8 * i)
							inpos += 1
					pixels = 0
					for i in range(8):
						pixels |= int(input[inpos]) << (8 * i)
						inpos += 1
					diff = (pixels >> 33) & 1
					horizontal = (pixels >> 32) & 1
					table1 = ETC1_MODIFIERS[(pixels >> 37) & 7]
					table2 = ETC1_MODIFIERS[(pixels >> 34) & 7]
					if diff:
						r = (pixels >> 59) & 0x1F
						g = (pixels >> 51) & 0x1F
						b = (pixels >> 43) & 0x1F
						color1[0] = (r << 3) | ((r >> 2) & 7)
						color1[1] = (g << 3) | ((g >> 2) & 7)
						color1[2] = (b << 3) | ((b >> 2) & 7)
						r += ETC1DiffComplement((pixels >> 56) & 7, 3)
						g += ETC1DiffComplement((pixels >> 48) & 7, 3)
						b += ETC1DiffComplement((pixels >> 40) & 7, 3)
						color2[0] = (r << 3) | ((r >> 2) & 7)
						color2[1] = (g << 3) | ((g >> 2) & 7)
						color2[2] = (b << 3) | ((b >> 2) & 7)
					else:
						color1[0] = ((pixels >> 60) & 0x0f) * 0x11
						color1[1] = ((pixels >> 52) & 0x0f) * 0x11
						color1[2] = ((pixels >> 44) & 0x0f) * 0x11
						color2[0] = ((pixels >> 56) & 0x0f) * 0x11
						color2[1] = ((pixels >> 48) & 0x0f) * 0x11
						color2[2] = ((pixels >> 40) & 0x0f) * 0x11
					#print(color1, color2, '%016X %016X' % (pixels, alphas))
					amounts = pixels & 0xFFFF
					signs = (pixels >> 16) & 0xFFFF
					for ypix in range(4):
						for xpix in range(4):
							x = xpix + xblock * 4 + xtile * 8
							y = ypix + yblock * 4 + ytile * 8
							if x >= width or y >= height:
								continue
							offset = (xpix * 4) + ypix
							if horizontal:
								table = table1 if ypix < 2 else table2
								color = color1 if ypix < 2 else color2
							else:
								table = table1 if xpix < 2 else table2
								color = color1 if xpix < 2 else color2
							amount = table[(amounts >> offset) & 1]
							sign = (signs >> offset) & 1
							if sign:
								amount = -amount
							outpos = (y * width + x) * 4
							output[outpos] = max(min(color[0] + amount, 0xFF), 0)
							output[outpos + 1] = max(min(color[1] + amount, 0xFF), 0)
							output[outpos + 2] = max(min(color[2] + amount, 0xFF), 0)
							output[outpos + 3] = ((alphas >> (offset * 4)) & 0x0F) * 0x11

def _extractBC4Texture(input, output, width, height, format, swizzlesize, littleendian):
	twidth = int(np.floor((width + 3) / 4))
	theight = int(np.floor((height + 3) / 4))
	lums = np.zeros(8, dtype=np.uint8)
	#int offset = 0;
	swizzle = Swizzle(twidth, BC4_BYTESPERTEXEL, swizzlesize);
	for ytile in range(theight):
		for xtile in range(twidth):
			offset = swizzle.getSwizzleOffset(xtile, ytile)
			lums[0] = input[offset]
			lums[1] = input[offset + 1]
			for i in range(2, 8):
				if lums[0] > lums[1]:
					lums[i] = ((8 - i) * lums[0] + (i - 1) * lums[1]) / 7
				elif i < 6:
					lums[i] = ((6 - i) * lums[0] + (i - 1) * lums[1]) / 7
				elif i == 6:
					lums[i] = 0;
				elif i == 7:
					lums[i] = 0xFF
			if littleendian:
				pixels1 = input[offset + 2] | (input[offset + 3] << 8) | (input[offset + 4] << 16)
				pixels2 = input[offset + 5] | (input[offset + 6] << 8) | (input[offset + 7] << 16)
			else:
				pixels1 = input[offset + 4] | (input[offset + 3] << 8) | (input[offset + 2] << 16)
				pixels2 = input[offset + 7] | (input[offset + 6] << 8) | (input[offset + 5] << 16)
			for ypix in range(4):
				for xpix in range(4):
					ypos = ytile * 4 + ypix
					xpos = xtile * 4 + xpix
					outpos = (ypos * width + xpos) * 4
					codeindex = ypix * 4 + xpix
					if codeindex < 8:
						lum = lums[(pixels1 >> (codeindex * 3)) & 7]
					else:
						lum = lums[(pixels2 >> ((codeindex - 8) * 3)) & 7]
					output[outpos] = output[outpos + 1] = output[outpos + 2] = lum
					output[outpos + 3] = 0xFF
	
def extractTiledTexture(input, output, width, height, format, swizzlesize, littleendian):
	if format in (ETC1, ETC1A4):
		_extractETC1Texture(input, output, width, height, format)
	elif format in (BC4, BC4_SNORM):
		_extractBC4Texture(input, output, width, height, format, swizzlesize, littleendian)
	else:
		_extractTiledTexture(input, output, width, height, format, littleendian)