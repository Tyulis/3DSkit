#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define bool uint8_t
#define true 1
#define false 0

#include "audio.h"
#include "compression.h"

static PyMethodDef functions[] = {
	{"decodeDSPADPCMblock", decodeDSPADPCMblock, METH_VARARGS, "Decodes a DSPADPCM sample block into 16-bits PCM data"},
	{"encodeDSPADPCMchannel", encodeDSPADPCMchannel, METH_VARARGS, "Encode a 16-bits PCM channel into DSP ADPCM"},
	{"generateDSPADPCMcoefs", generateDSPADPCMcoefs, METH_VARARGS, "Generate coefficients for DSPADPCM encoding"},
	{"compressLZ11", compressLZ11, METH_VARARGS, "Compress a byte stream in LZ11"},
	{"decompressLZ11", decompressLZ11, METH_VARARGS, "Decompress a LZ11 compressed stream"},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef c3DSkitdef = {
	PyModuleDef_HEAD_INIT,
	"c3DSkit", "A C module to make 3DSkit faster",
	-1, functions
};

PyMODINIT_FUNC PyInit_c3DSkit(void){
	import_array();
	srand(time(NULL));
	return PyModule_Create(&c3DSkitdef);
}