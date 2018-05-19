#ifndef _C3DSKIT_GRAPHICS_H_
#define _C3DSKIT_GRAPHICS_H_

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

PyObject* extractTiledTexture(PyObject* self, PyObject* args);
PyObject* getTextureFormatId(PyObject* self, PyObject* args);

#include "graphics.c"

#endif