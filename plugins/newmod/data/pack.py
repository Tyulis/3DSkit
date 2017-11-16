# -*- coding:utf-8 -*-
from util import error
from util.funcops import ClsFunc
from util.rawutil import TypeWriter
from util.fileops import *

class pack{{FORMAT}}(ClsFunc, TypeWriter):
	def main(self, filenames, outname, endian, verbose, opts={}):
		self.byteorder = endian
		self.verbose = verbose
>f		inname = filenames[0]
