#ifndef _C3DSKIT_COMPRESSION_H_
#define _C3DSKIT_COMPRESSION_H_

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <stdint.h>

PyObject* compressLZ11(PyObject* self, PyObject* args);

#include "compression.c"

#endif