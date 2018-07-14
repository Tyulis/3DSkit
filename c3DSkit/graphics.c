#include "graphics.h"
#include <stdio.h>

#define TESTNAME(name, value, id) if(strcmp(name,value)==0){return Py_BuildValue("i",id);}
#define TESTPXSIZE(format, id, pxsize) if(format==id){return pxsize;}

#define BC4_BYTESPERTEXEL 8

enum TEXTURE_FORMATS{
	RGBA8, RGB8, RGBA5551, RGB565, RGBA4,
	LA8, RG8, L8, A8, LA4, L4, A4,
	ETC1, ETC1A4, BC1, BC2, BC3, BC4, BC5, BC6H, BC7,
	RGBA8_SRGB, BC1_SRGB, BC2_SRGB, BC3_SRGB,
	BC4_SNORM, BC5_SNORM, BC6H_SF16, BC7_SRGB,
};

uint8_t ETC1_MODIFIERS[8][2] = {
	{2, 8}, {5, 17}, {9, 29}, {13, 42},
	{18, 60}, {24, 80}, {33, 106}, {47, 183}
};

// Swizzling is taken from gdkchan's BnTxx

typedef struct {
	int bhshift;
	int bppshift;
	int bhmask;
	int xshift;
	int gobstride;
} Swizzle;

static int _getShift(int value){
	if (value == 0) return 0;
	int shift = 0;
	while (((value >> shift) & 1) == 0){ shift++; }
	return shift;
}

static void _makeSwizzle(Swizzle* swizzle, int width, int bpp, int blockheight){
	if (blockheight <= 0)blockheight = 16;
	swizzle->bhmask = blockheight * 8 - 1;
	swizzle->bhshift = _getShift(blockheight * 8);
	swizzle->bppshift = _getShift(bpp);
	swizzle->gobstride = 512 * blockheight * (int)ceil(width * bpp / 64.0);
	swizzle->xshift = _getShift(512 * blockheight);
}

static int _getSwizzleOffset(Swizzle* swizzle, int x, int y){
	x <<= swizzle->bppshift;
	int pos = (y >> swizzle->bhshift) * swizzle->gobstride;
	pos += (x >> 6) << swizzle->xshift;
	pos += ((y & swizzle->bhmask) >> 3) << 9;
	pos += ((x & 0x3f) >> 5) << 8;
	pos += ((y & 0x07) >> 1) << 6;
	pos += ((x & 0x1f) >> 4) << 5;
	pos += ((y & 0x01) >> 0) << 4;
	pos += ((x & 0x0f) >> 0) << 0;
	return pos;
}
	

static int getPixelSize(int format){
	TESTPXSIZE(format, RGBA8, 4);
	TESTPXSIZE(format, RGB8, 3);
	TESTPXSIZE(format, RGBA5551, 2);
	TESTPXSIZE(format, RGB565, 2);
	TESTPXSIZE(format, RGBA4, 2);
	TESTPXSIZE(format, LA8, 2);
	TESTPXSIZE(format, RG8, 2);
	TESTPXSIZE(format, L8, 1);
	TESTPXSIZE(format, A8, 1);
	TESTPXSIZE(format, LA4, 1);
	return -1;
}

static void _extractTiledTexture(uint8_t* input, uint8_t* output, int width, int height, int format, bool littleendian){
	int tilesx = (int)ceil((double)width / 8);
	int tilesy = (int)ceil((double)height / 8);
	int datawidth = 1 << (int)ceil(LOG2((double)width));
	//int dataheight = 1 << (int)ceil(LOG2((double)height));
	int totalx = (int)ceil((double)datawidth / 8.0);
	//int totaly = (int)ceil((double)dataheight / 8.0);
	int pxsize = getPixelSize(format);
	for (int ytile = 0; ytile < tilesy; ytile++){
		for (int xtile = 0; xtile < tilesx; xtile++){
			for (int ysub = 0; ysub < 2; ysub++){
				for (int xsub = 0; xsub < 2; xsub++){
					for (int yblock = 0; yblock < 2; yblock++){
						for (int xblock = 0; xblock < 2; xblock++){
							for (int ypix = 0; ypix < 2; ypix++){
								for (int xpix = 0; xpix < 2; xpix++){
									int ypos = ytile * 8 + ysub * 4 + yblock * 2 + ypix;
									int xpos = xtile * 8 + xsub * 4 + xblock * 2 + xpix;
									if (xpos >= width || ypos >= height){
										continue;
									}
									int outpos = (ypos * width + xpos) * 4;
									uint8_t r = 0, g = 0, b = 0, a = 0;
									if (format == L4 || format == A4){
										int shift = xpix * 4;
										int inpos = ytile * totalx * 32 + xtile * 32 + ysub * 16 + xsub * 8 + yblock * 4 + xblock * 2 + ypix;
										uint8_t byte = input[inpos];
										if (format == L4){
											r = g = b = ((byte >> shift) & 0x0F) * 0x11;
											a = 0xFF;
										} else if (format == A4){
											r = g = b = 0xFF;
											a = ((byte >> shift) & 0x0F) * 0x11;
										}
									} else {
										int inpos = (ytile * totalx * 64 + xtile * 64 + ysub * 32 + xsub * 16 + yblock * 8 + xblock * 4 + ypix * 2 + xpix) * pxsize;
										if (littleendian){
											if (format == RGBA8){
												r = input[inpos + 3];
												g = input[inpos + 2];
												b = input[inpos + 1];
												a = input[inpos];
											} else if (format == RGB8){
												r = input[inpos + 2];
												g = input[inpos + 1];
												b = input[inpos];
												a = 0xFF;
											} else if (format == RGBA5551){
												r = (input[inpos + 1] >> 3) * 8.225806451612;
												g = (((input[inpos + 1] & 0x07) << 2) | (input[inpos] >> 6)) * 8.225806451612;
												b = ((input[inpos] >> 1) & 0x1F) * 8.225806451612;
												a = (input[inpos] & 1) * 0xFF;
											} else if (format == RGB565){
												r = (input[inpos + 1] >> 3) * 8.225806451612;
												g = (((input[inpos + 1] & 0x07) << 3) | (input[inpos] >> 5)) * 4.0476190476190;
												b = (input[inpos] & 0x1F) * 8.225806451612;
												a = 0xFF;
											} else if (format == RGBA4){
												r = (input[inpos + 1] >> 4) * 0x11;
												g = (input[inpos + 1] & 0x0F) * 0x11;
												b = (input[inpos] >> 4) * 0x11;
												a = (input[inpos] & 0x0F) * 0x11;
											} else if (format == LA8){
												r = g = b = input[inpos + 1];
												a = input[inpos];
											} else if (format == RG8){
												r = input[inpos + 1];
												g = input[inpos];
												b = 0;
												a = 0xFF;
											} else if (format == L8){
												r = g = b = input[inpos];
												a = 0xFF;
											} else if (format == A8){
												r = g = b = 0xFF;
												a = input[inpos];
											} else if (format == LA4){
												r = g = b = (input[inpos] >> 4) * 0x11;
												a = (input[inpos] & 0x0F) * 0x11;
											}
										} else {
											if (format == RGBA8){
												r = input[inpos];
												g = input[inpos + 1];
												b = input[inpos + 2];
												a = input[inpos + 3];
											} else if (format == RGB8){
												r = input[inpos];
												g = input[inpos + 1];
												b = input[inpos + 2];
												a = 0xFF;
											} else if (format == RGBA5551){
												r = (input[inpos] >> 3) * 8.225806451612;
												g = (((input[inpos] & 0x07) << 2) | (input[inpos + 1] >> 6)) * 8.225806451612;
												b = ((input[inpos + 1] >> 1) & 0x1F) * 8.225806451612;
												a = (input[inpos + 1] & 1) * 0xFF;
											} else if (format == RGB565){
												r = (input[inpos] >> 3) * 8.225806451612;
												g = (((input[inpos] & 0x07) << 3) | (input[inpos + 1] >> 5)) * 4.0476190476190;
												b = (input[inpos + 1] & 0x1F) * 8.225806451612;
												a = 0xFF;
											} else if (format == RGBA4){
												r = (input[inpos] >> 4) * 0x11;
												g = (input[inpos] & 0x0F) * 0x11;
												b = (input[inpos + 1] >> 4) * 0x11;
												a = (input[inpos + 1] & 0x0F) * 0x11;
											} else if (format == LA8){
												r = g = b = input[inpos];
												a = input[inpos + 1];
											} else if (format == RG8){
												r = input[inpos];
												g = input[inpos + 1];
												b = 0;
												a = 0xFF;
											} else if (format == L8){
												r = g = b = input[inpos];
												a = 0xFF;
											} else if (format == A8){
												r = g = b = 0xFF;
												a = input[inpos];
											} else if (format == LA4){
												r = g = b = (input[inpos] >> 4) * 0x11;
												a = (input[inpos] & 0x0F) * 0x11;
											}
										}
									}
									output[outpos] = r;
									output[outpos + 1] = g;
									output[outpos + 2] = b;
									output[outpos + 3] = a;
								}
							}
						}
					}
				}
			}
		}
	}
}

//ETC1 decompression is inspirated from ObsidianX's 3dstools BFLIM implementation

static int ETC1DiffComplement(int val, int bits){
	if ((val >> (bits - 1)) == 0){
		return val;
	}
	return val - (1 << bits);
}

static void _extractETC1Texture(uint8_t* input, uint8_t* output, int width, int height, int format){
	bool hasalpha = (format == ETC1A4);
	int tilew = 1 << (int)ceil(LOG2(width / 8));
	int tileh = 1 << (int)ceil(LOG2(height / 8));
	int inpos = 0;
	uint64_t alphas = 0xFFFFffffFFFFffff;
	uint8_t color1[3], color2[3];
	uint8_t r, g, b;
	for (int ytile = 0; ytile < tileh; ytile++){
		for (int xtile = 0; xtile < tilew; xtile++){
			for (int yblock = 0; yblock < 2; yblock++){
				for (int xblock = 0; xblock < 2; xblock++){
					if (hasalpha){
						alphas = 0;
						for (int i = 0; i < 8; i++){
							alphas |= (uint64_t)input[inpos++] << (8 * i);
						}
					}
					uint64_t pixels = 0;
					for (int i = 0; i < 8; i++){
						pixels |= (uint64_t)input[inpos++] << (8 * i);
					}
					bool diff = (pixels >> 33) & 1;
					bool horizontal = (pixels >> 32) & 1;
					uint8_t* table1 = ETC1_MODIFIERS[(pixels >> 37) & 7];
					uint8_t* table2 = ETC1_MODIFIERS[(pixels >> 34) & 7];
					if (diff){
						r = (pixels >> 59) & 0x1F;
						g = (pixels >> 51) & 0x1F;
						b = (pixels >> 43) & 0x1F;
						color1[0] = (r << 3) | ((r >> 2) & 7);
						color1[1] = (g << 3) | ((g >> 2) & 7);
						color1[2] = (b << 3) | ((b >> 2) & 7);
						r += ETC1DiffComplement((pixels >> 56) & 7, 3);
						g += ETC1DiffComplement((pixels >> 48) & 7, 3);
						b += ETC1DiffComplement((pixels >> 40) & 7, 3);
						color2[0] = (r << 3) | ((r >> 2) & 7);
						color2[1] = (g << 3) | ((g >> 2) & 7);
						color2[2] = (b << 3) | ((b >> 2) & 7);
					} else {
						color1[0] = ((pixels >> 60) & 0x0f) * 0x11;
						color1[1] = ((pixels >> 52) & 0x0f) * 0x11;
						color1[2] = ((pixels >> 44) & 0x0f) * 0x11;
						color2[0] = ((pixels >> 56) & 0x0f) * 0x11;
						color2[1] = ((pixels >> 48) & 0x0f) * 0x11;
						color2[2] = ((pixels >> 40) & 0x0f) * 0x11;
					}
					uint16_t amounts = pixels & 0xFFFF;
					uint16_t signs = (pixels >> 16) & 0xFFFF;
					for (int ypix = 0; ypix < 4; ypix++){
						for (int xpix = 0; xpix < 4; xpix++){
							int x = xpix + xblock * 4 + xtile * 8;
							int y = ypix + yblock * 4 + ytile * 8;
							if (x >= width){
								continue;
							} else if (y >= height){
								continue;
							}
							int offset = xpix * 4 + ypix;
							uint8_t* table;
							uint8_t* color;
							if (horizontal){
								table = (ypix < 2) ? table1 : table2;
								color = (ypix < 2) ? color1 : color2;
							} else {
								table = (xpix < 2) ? table1 : table2;
								color = (xpix < 2) ? color1 : color2;
							}
							int amount = table[(amounts >> offset) & 1];
							int sign = (signs >> offset) & 1;
							if (sign == 1){
								amount = -amount;
							}
							int outpos = (y * width + x) * 4;
							output[outpos] = MAX(MIN(color[0] + amount, 0xFF), 0);
							output[outpos + 1] = MAX(MIN(color[1] + amount, 0xFF), 0);
							output[outpos + 2] = MAX(MIN(color[2] + amount, 0xFF), 0);
							output[outpos + 3] = ((alphas >> (offset * 4)) & 0x0F) * 0x11;
						}
					}
				}
			}
		}
	}
}

static void _extractBC4Texture(uint8_t* input, uint8_t* output, int width, int height, int format, int swizzlesize, bool littleendian){
	int twidth = (width + 3) / 4;
	int theight = (height + 3) / 4;
	uint8_t lums[8];
	Swizzle swizzle;
	//int offset = 0;
	_makeSwizzle(&swizzle, twidth, BC4_BYTESPERTEXEL, swizzlesize);
	for (int ytile = 0; ytile < theight; ytile++){
		for (int xtile = 0; xtile < twidth; xtile++){
			int offset = _getSwizzleOffset(&swizzle, xtile, ytile);
			lums[0] = input[offset];
			lums[1] = input[offset + 1];
			for (int i = 2; i < 8; i++){
				if (lums[0] > lums[1]){
					lums[i] = (uint8_t)(((8 - i) * lums[0] + (i - 1) * lums[1]) / 7);
				} else if (i < 6){
					lums[i] = (uint8_t)(((6 - i) * lums[0] + (i - 1) * lums[1]) / 7);
				} else if (i == 6){
					lums[i] = 0;
				} else if (i == 7){
					lums[i] = 0xFF;
				}
			}
			int pixels1, pixels2;
			if (littleendian){
				pixels1 = input[offset + 2] | (input[offset + 3] << 8) | (input[offset + 4] << 16);
				pixels2 = input[offset + 5] | (input[offset + 6] << 8) | (input[offset + 7] << 16);
			} else {
				pixels1 = input[offset + 4] | (input[offset + 3] << 8) | (input[offset + 2] << 16);
				pixels2 = input[offset + 7] | (input[offset + 6] << 8) | (input[offset + 5] << 16);
			}
			int lum;
			for (int ypix = 0; ypix < 4; ypix++){
				for (int xpix = 0; xpix < 4; xpix++){
					int ypos = ytile * 4 + ypix;
					int xpos = xtile * 4 + xpix;
					int outpos = (ypos * width + xpos) * 4;
					int codeindex = ypix * 4 + xpix;
					if (codeindex < 8){
						lum = lums[(pixels1 >> (codeindex * 3)) & 7];
					} else {
						lum = lums[(pixels2 >> ((codeindex - 8) * 3)) & 7];
					}
					output[outpos] = output[outpos + 1] = output[outpos + 2] = lum;
					output[outpos + 3] = 0xFF;
				}
			}
		}
	}
}

PyObject* extractTiledTexture(PyObject* self, PyObject* args){
	PyArrayObject* input_obj;
	PyArrayObject* output_obj;
	int width, height, format, littleendian, swizzlesize;
	if (!PyArg_ParseTuple(args, "O!O!iiiii", &PyArray_Type, &input_obj, &PyArray_Type, &output_obj, &width, &height, &format, &swizzlesize, &littleendian)){
		return NULL;
	}
	uint8_t* input = (uint8_t*)PyArray_DATA(input_obj);
	uint8_t* output = (uint8_t*)PyArray_DATA(output_obj);
	if (format == ETC1 || format == ETC1A4){
		_extractETC1Texture(input, output, width, height, format);
	} else if (format == BC4){
		_extractBC4Texture(input, output, width, height, format, swizzlesize, (bool)littleendian);
	} else {
		_extractTiledTexture(input, output, width, height, format, (bool)littleendian);
	}
	return Py_BuildValue("i", 0);
}

PyObject* getTextureFormatId(PyObject* self, PyObject* args){
	const char* name;
	if (!PyArg_ParseTuple(args, "s", &name)){
		return NULL;
	}
	TESTNAME(name, "RGBA8", RGBA8);
	TESTNAME(name, "RGB8", RGB8);
	TESTNAME(name, "RGBA5551", RGBA5551);
	TESTNAME(name, "RGB565", RGB565);
	TESTNAME(name, "RGBA4", RGBA4);
	TESTNAME(name, "LA8", LA8);
	TESTNAME(name, "RG8", RG8);
	TESTNAME(name, "L8", L8);
	TESTNAME(name, "A8", A8);
	TESTNAME(name, "LA4", LA4);
	TESTNAME(name, "L4", L4);
	TESTNAME(name, "A4", A4);
	TESTNAME(name, "ETC1", ETC1);
	TESTNAME(name, "ETC1A4", ETC1A4);
	//TESTNAME(name, "BC1", BC1);
	//TESTNAME(name, "BC2", BC2);
	//TESTNAME(name, "BC3", BC3);
	TESTNAME(name, "BC4", BC4);
	//TESTNAME(name, "BC5", BC5);
	//TESTNAME(name, "BC6H", BC6H);
	//TESTNAME(name, "BC7", BC7);
	//TESTNAME(name, "RGBA8_SRGB", RGBA8_SRGB);
	//TESTNAME(name, "BC1_SRGB", BC1_SRGB);
	//TESTNAME(name, "BC2_SRGB", BC2_SRGB);
	//TESTNAME(name, "BC3_SRGB", BC3_SRGB);
	//TESTNAME(name, "BC4_SNORM", BC4_SNORM);
	//TESTNAME(name, "BC5_SNORM", BC5_SNORM);
	//TESTNAME(name, "BC6H_SF16", BC6H_SF16);
	//TESTNAME(name, "BC7_SRGB", BC7_SRGB);
	return Py_BuildValue("I", -1);
}
