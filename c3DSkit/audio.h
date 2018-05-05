#ifndef _C3DSKIT_AUDIO_H_
#define _C3DSKIT_AUDIO_H_

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <stdint.h>

PyObject* decodeDSPADPCMblock(PyObject* self, PyObject* args);

#include "audio.c"

#endif