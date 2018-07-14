#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define bool uint8_t
#define true 1
#define false 0

#define ABS(val) ((val < 0) ? -val : val)
#define LOG2(val) (log(val) / log(2))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define CLAMP(val, low, high) ((val < low) ? low : ((val > high) ? high : val))

#include "audio.h"
#include "compression.h"
#include "graphics.h"

static PyObject* _confirm(PyObject* self, PyObject* args){
	return Py_BuildValue("i", 1);
}

static PyMethodDef functions[] = {
	{"_confirm", _confirm, METH_VARARGS, "Confirms the existence of c3DSkit"},
	{"decodeDSPADPCMblock", decodeDSPADPCMblock, METH_VARARGS, "Decodes a DSPADPCM sample block into 16-bits PCM data"},
	{"encodeDSPADPCMchannel", encodeDSPADPCMchannel, METH_VARARGS, "Encode a 16-bits PCM channel into DSP ADPCM"},
	{"generateDSPADPCMcoefs", generateDSPADPCMcoefs, METH_VARARGS, "Generate coefficients for DSPADPCM encoding"},
	{"compressLZ11", compressLZ11, METH_VARARGS, "Compress a byte stream in LZ11"},
	{"decompressLZ11", decompressLZ11, METH_VARARGS, "Decompress a LZ11 compressed stream"},
	{"extractTiledTexture", extractTiledTexture, METH_VARARGS, "Extracts a tiled texture (like BFLIM ones) to an RGBA byte array"},
	{"getTextureFormatId", getTextureFormatId, METH_VARARGS, "Returns the internal format ID from its name"},
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