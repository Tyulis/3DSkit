#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <stdint.h>

static int* _decodeDSPADPCMblock(uint8_t* adpcm, int16_t* pcmout, int16_t* coefs, int samplecount, int blockstart, int last1, int last2){
	int framecount = samplecount / 14;
	int framestart = 0;
	int sampleidx = 0;
	int16_t sample = 0;
	for (int i = 0; i < framecount; i++){
		uint8_t info = adpcm[framestart];
		int shift = (info & 0x0f) + 11;
		int coefindex = info >> 4;
		int16_t coef1 = coefs[coefindex * 2];
		int16_t coef2 = coefs[coefindex * 2 + 1];
		for (int byteidx = 0; byteidx < 7; byteidx++){
			uint8_t byte = adpcm[framestart + byteidx];
			int high = byte >> 4;
			int low = byte & 0x0f;
			if (high > 7){
				high -= 16;
			}
			if (low > 7){
				low -= 16;
			}
			sample = ((high << shift) + coef1 * last1 + coef2 * last2) / 2048;
			pcmout[blockstart + sampleidx] = sample;
			last2 = last1;
			last1 = sample;
			sampleidx += 1;
			sample = ((low << shift) + coef1 * last1 + coef2 * last2) / 2048;
			pcmout[blockstart + sampleidx] = sample;
			last2 = last1;
			last1 = sample;
			sampleidx += 1;
		}
		framestart += 8;
	}
	int* lasts = (int*)calloc(sizeof(int), 2);
	lasts[0] = last1;
	lasts[1] = last2;
	return lasts;
}

static PyObject* decodeDSPADPCMblock(PyObject* self, PyObject* args){
	PyArrayObject* adpcm_obj;
	PyArrayObject* pcmout_obj;
	PyArrayObject* coefs_obj;
	int samplecount;
	int blockstart;
	int last1;
	int last2;
	if (!PyArg_ParseTuple(args, "O!O!O!iiii", &PyArray_Type, &adpcm_obj, &PyArray_Type, &pcmout_obj, &PyArray_Type, &coefs_obj, &samplecount, &blockstart, &last1, &last2)){
		return NULL;
	}
	uint8_t* adpcm = (uint8_t*)PyArray_DATA(adpcm_obj);
	int16_t* pcmout = (int16_t*)PyArray_DATA(pcmout_obj);
	int16_t* coefs = (int16_t*)PyArray_DATA(coefs_obj);
	int* lasts = _decodeDSPADPCMblock(adpcm, pcmout, coefs, samplecount, blockstart, last1, last2);
	PyObject* result = Py_BuildValue("(ii)", lasts[0], lasts[1]);
	free(lasts);
	return result;
}

static PyMethodDef functions[] = {
	{"decodeDSPADPCMblock", decodeDSPADPCMblock, METH_VARARGS, "Decodes a DSPADPCM sample block into 16-bits PCM data"},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef c3DSkitdef = {
	PyModuleDef_HEAD_INIT,
	"c3DSkit", "A C module to make 3DSkit faster",
	-1, functions
};

PyMODINIT_FUNC PyInit_c3DSkit(void){
	import_array();
	return PyModule_Create(&c3DSkitdef);
}