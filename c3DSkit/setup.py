import numpy
from setuptools import setup, Extension

c3DSkit = Extension('c3DSkit', sources=['c3DSkit.c'], include_dirs=[numpy.get_include()])

setup(
	name = 'c3DSkit',
	version = '1.4',
	ext_modules = [c3DSkit],
	url = 'https://github.com/Tyulis/3DSkit',
	author = 'Tyulis',
)