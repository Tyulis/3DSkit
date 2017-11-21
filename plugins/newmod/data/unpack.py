# -*- coding:utf-8 -*-
import os
from util import error
>afrom util.funcops import byterepr
>ffrom util.funcops import byterepr, ClsFunc
from util.fileops import *
import util.rawutil as rawutil

>aclass extract{{FORMAT}} (rawutil.TypeReader):
>fclass extract{{FORMAT}} (rawutil.TypeReader, ClsFunc):
>a	def __init__(self, filename, data, verbose, opts={}):
>a		self.outdir = make_outdir(filename)
>a		self.verbose = verbose
>a		#Place your code here
>f	def main(self, filename, data, verbose, opts={}):
>f		self.outfile = make_outfile(filename, '{{OUTEXT}}')
>f		self.verbose = verbose
>f		#Place your code here
>a
>a	def extract(self):
>a		#Code to really extract files
		pass
