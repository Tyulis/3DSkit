#include "graphics.h"

#define TESTNAME(name, value, id) if(strcmp(name,value)==0){return Py_BuildValue("i",id);}
#define TESTPXSIZE(format, id, pxsize) if(format==id){return pxsize;}

enum TEXTURE_FORMATS{
	RGBA8, RGB8, RGBA5551, RGB565, RGBA4,
	LA8, HILO8, L8, A8, LA4, L4, A4,
	ETC1, ETC1A4, BC1, BC2, BC3, BC4, BC5,
	RGBA8_SRGB, BC1_SRGB, BC2_SRGB, BC3_SRGB
};

uint8_t ETC1_MODIFIERS[8][2] = {
	{2, 8}, {5, 17}, {9, 29}, {13, 42},
	{18, 60}, {24, 80}, {33, 106}, {47, 183}
};

static int getPixelSize(int format){
	TESTPXSIZE(format, RGBA8, 4);
	TESTPXSIZE(format, RGB8, 3);
	TESTPXSIZE(format, RGBA5551, 2);
	TESTPXSIZE(format, RGB565, 2);
	TESTPXSIZE(format, RGBA4, 2);
	TESTPXSIZE(format, LA8, 2);
	TESTPXSIZE(format, HILO8, 2);
	TESTPXSIZE(format, L8, 1);
	TESTPXSIZE(format, A8, 1);
	TESTPXSIZE(format, LA4, 1);
	return -1;
}

static void _extractTiledTexture(uint8_t* input, uint8_t* output, int width, int height, int format){
	int tilesx = (int)ceil((double)width / 8);
	int tilesy = (int)ceil((double)height / 8);
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
									int outpos = (ypos * width + xpos) * 4;
									uint8_t r = 0, g = 0, b = 0, a = 0;
									if (format == L4 || format == A4){
										int shift = xpix * 4;
										int inpos = ytile * tilesx * 32 + xtile * 32 + ysub * 16 + xsub * 8 + yblock * 4 + xblock * 2 + ypix;
										uint8_t byte = input[inpos];
										if (format == L4){
											r = g = b = ((byte >> shift) & 0x0F) * 0x11;
											a = 0xFF;
										} else if (format == A4){
											r = g = b = 0xFF;
											a = ((byte >> shift) & 0x0F) * 0x11;
										}
									} else {
										int inpos = (ytile * tilesx * 64 + xtile * 64 + ysub * 32 + xsub * 16 + yblock * 8 + xblock * 4 + ypix * 2 + xpix) * pxsize;
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
											r = (input[inpos] >> 3) * 8.225806451612904;
											g = (((input[inpos] & 0x07) << 2) | (input[inpos + 1] >> 6)) * 8.225806451612904;
											b = ((input[inpos + 1] >> 1) & 0x1F) * 8.225806451612904;
											a = (input[inpos + 1] & 1) * 0xFF;
										} else if (format == RGB565){
											r = (input[inpos] >> 3) * 8.225806451612904;
											g = (((input[inpos] & 0x07) << 3) | (input[inpos + 1] >> 5)) * 4.0476190476190474;
											b = (input[inpos + 1] & 0x1F) * 8.225806451612904;
											a = 0xFF;
										} else if (format == RGBA4){
											r = (input[inpos] >> 4) * 0x11;
											g = (input[inpos] & 0x0F) * 0x11;
											b = (input[inpos + 1] >> 4) * 0x11;
											a = (input[inpos + 1] & 0x0F) * 0x11;
										} else if (format == LA8){
											r = g = b = input[inpos];
											a = input[inpos + 1];
										} else if (format == HILO8){
											// ?
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
	for (int ytile = 0; ytile < tileh; ytile++){
		for (int xtile = 0; xtile < tilew; xtile++){
			for (int yblock = 0; yblock < 2; yblock++){
				for (int xblock = 0; xblock < 2; xblock++){
					if (hasalpha){
						alphas = 0;
						for (int i = 0; i < 8; i++){
							alphas |= input[inpos++] << i;
						}
					}
					uint64_t pixels = 0;
					for (int i = 0; i < 8; i++){
						pixels |= input[inpos++] << i;
					}
					bool diff = (pixels >> 33) & 1;
					bool horizontal = (pixels >> 32) & 1;
					uint8_t* table1 = ETC1_MODIFIERS[(pixels >> 37) & 7];
					uint8_t* table2 = ETC1_MODIFIERS[(pixels >> 34) & 7];
					uint8_t color1[3], color2[3];
					uint8_t r, g, b;
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
							int offset = xpix + 4 * ypix;
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

static void _extractBC4Texture(uint8_t* input, uint8_t* output, int width, int height, int format){
	int tilew = 1 << (int)ceil(LOG2(width / 8));
	int tileh = 1 << (int)ceil(LOG2(height / 8));
	uint8_t indices[16];
	uint8_t values[8];
	int inpos = 0;
	for (int ytile = 0; ytile < tileh; ytile++){
		for (int xtile = 0; xtile < tilew; xtile++){
			for (int yblock = 0; yblock < 2; yblock++){
				for (int xblock = 0; xblock < 2; xblock++){
					values[0] = input[inpos++];
					values[1] = input[inpos++];
					for (int i = 0; i < 16; i += 8){
						indices[i] = input[inpos + 2] & 0x07;
						indices[i + 1] = (input[inpos + 2] >> 3) & 0x07;
						indices[i + 2] = ((input[inpos + 1] & 1) << 2) | (input[inpos + 2] >> 6);
						indices[i + 3] = (input[inpos + 1] >> 1) & 0x07;
						indices[i + 4] = (input[inpos + 1] >> 4) & 0x07;
						indices[i + 5] = (input[inpos + 1] >> 7) | ((input[inpos] & 0x03) << 1);
						indices[i + 6] = (input[inpos] >> 2) & 0x07;
						indices[i + 7] = (input[inpos] >> 5) & 0x07;
						inpos += 3;
					}
					if (values[0] > values[1]){
						for (int i = 2; i < 8; i++){
							values[i] = ((8 - i) * values[0] + (i - 1) * values[1]) / 7;
						}
					} else {
						for (int i = 2; i < 6; i++){
							values[i] = ((6 - i) * values[0] + (i - 1) * values[1]) / 7;
						}
						values[6] = 0;
						values[7] = 0xFF;
					}
					for (int ypix = 0; ypix < 4; ypix++){
						for (int xpix = 0; xpix < 4; xpix++){
							int xpos = xtile * 8 + xblock * 4 + xpix;
							int ypos = ytile * 8 + yblock * 4 + ypix;
							int outpos = (ypos * width + xpos) * 4;
							int index = ypix * 4 + xpix;
							output[outpos] = output[outpos + 1] = output[outpos + 2] = values[indices[index]];
							output[outpos + 3] = 0xFF;
						}
					}
				}
			}
		}
	}
}

PyObject* extractTiledTexture(PyObject* self, PyObject* args){
	PyArrayObject* input_obj;
	PyArrayObject* output_obj;
	int width;
	int height;
	int format;
	if (!PyArg_ParseTuple(args, "O!O!iii", &PyArray_Type, &input_obj, &PyArray_Type, &output_obj, &width, &height, &format)){
		return NULL;
	}
	uint8_t* input = (uint8_t*)PyArray_DATA(input_obj);
	uint8_t* output = (uint8_t*)PyArray_DATA(output_obj);
	if (format == ETC1 || format == ETC1A4){
		_extractETC1Texture(input, output, width, height, format);
	} else if (format == BC4){
		_extractBC4Texture(input, output, width, height, format);
	} else {
		_extractTiledTexture(input, output, width, height, format);
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
	TESTNAME(name, "HILO8", HILO8);
	TESTNAME(name, "L8", L8);
	TESTNAME(name, "A8", A8);
	TESTNAME(name, "LA4", LA4);
	TESTNAME(name, "L4", L4);
	TESTNAME(name, "A4", A4);
	TESTNAME(name, "ETC1", ETC1);
	TESTNAME(name, "ETC1A4", ETC1A4);
	TESTNAME(name, "BC1", BC1);
	TESTNAME(name, "BC2", BC2);
	TESTNAME(name, "BC3", BC3);
	TESTNAME(name, "BC4", BC4);
	TESTNAME(name, "BC5", BC5);
	TESTNAME(name, "RGBA8_SRGB", RGBA8_SRGB);
	TESTNAME(name, "BC1_SRGB", BC1_SRGB);
	TESTNAME(name, "BC2_SRGB", BC2_SRGB);
	TESTNAME(name, "BC3_SRGB", BC3_SRGB);
	return Py_BuildValue("I", 0xFF);
}