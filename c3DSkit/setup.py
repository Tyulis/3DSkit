from distutils.core import setup, Extension
try:
	import numpy
except ImportError:
	import sys
	print("3DSkit requires numpy to work properly. You can install it by typing pip install numpy")
	sys.exit(1)

c3DSkit = Extension('c3DSkit', sources=['c3DSkit.c'], include_dirs=[numpy.get_include()])

setup(
	name = 'c3DSkit',
	version = '1.1',
	ext_modules = [c3DSkit],
	url = 'https://github.com/Tyulis/3DSkit',
	author = 'Tyulis',
)