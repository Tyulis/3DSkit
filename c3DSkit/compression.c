#include "compression.h"
#include <stdio.h>

// Near full transcription of the python algorithm, with pointers. Many pointers
static int _compressLZ11(uint8_t* input, uint8_t* output, int insize){
	uint8_t* outptr = output + 1;
	uint8_t* header = output;
	uint8_t* sub = NULL;
	uint8_t* refbyte = NULL;
	int outsize = 0;
	int pos = 0;
	int size = 0;
	int count = 0;
	int disp = 0;
	int refpos = 0;
	int blockidx = 0;
	int windowstart = 0;
	while (pos < insize){
		if (blockidx >= 8){
			header = outptr;
			blockidx = 0;
			outptr++;
			outsize++;
		}
		if (insize - pos <= 3){  //3 last bytes, not enough data
			*outptr++ = input[pos++];
			blockidx++;
			outsize++;
			continue;
		}
		windowstart = pos - 4096;
		if (windowstart < 0){
			windowstart = 0;
		}
		sub = &input[pos];
		refpos = -1;
		for (int i = windowstart; i < pos; i++){
			if (sub[0] == input[i] && sub[1] == input[i + 1] && sub[2] == input[i + 2]){
				refpos = i;
				break;
			}
		}
		if (refpos == -1){
			*outptr++ = input[pos++];
			outsize++;
		} else {
			size = 3;
			disp = pos - refpos - 1;
			pos += 3;
			refbyte = &input[refpos + 3];
			while(*refbyte == input[pos] && pos < insize && size < 0x10110){
				refbyte++;
				pos++;
				size++;
			}
			if (size > 0xff + 0x11){
				count = size - 0x111;
				*outptr++ = 0x10 | (count >> 12);
				*outptr++ = (count >> 4) & 0xff;
				*outptr++ = ((count & 0x0f) << 4) | (disp >> 8);
				*outptr++ = disp & 0xff;
				outsize += 4;
			} else if (size > 0xf + 1){
				count = size - 0x11;
				*outptr++ = count >> 4;
				*outptr++ = ((count & 0x0f) << 4) | (disp >> 8);
				*outptr++ = disp & 0xff;
				outsize += 3;
			} else {
				count = size - 1;
				*outptr++ = (count << 4) | (disp >> 8);
				*outptr++ = disp & 0xff;
				outsize += 2;
			}
			*header |= 1 << (7 - blockidx);
		}
		blockidx++;
	}
	return outsize + 1;
}

PyObject* compressLZ11(PyObject* self, PyObject* args){
	PyArrayObject* input_obj;
	PyArrayObject* output_obj;
	int insize;
	if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &input_obj, &PyArray_Type, &output_obj, &insize)){
		return NULL;
	}
	uint8_t* input = (uint8_t*)PyArray_DATA(input_obj);
	uint8_t* output = (uint8_t*)PyArray_DATA(output_obj);
	int outsize = _compressLZ11(input, output, insize);
	return Py_BuildValue("i", outsize);
}