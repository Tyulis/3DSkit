# -*- coding:utf-8 -*-
import os
from util import error
from util.funcops import byterepr
from util.fileops import *
import util.rawutil as rawutil

class extractRomFS (rawutil.TypeReader):
	def __init__(self, filename, data, opts={}):
		self.outdir = make_outdir(filename)
		#Place your code here

	def extract(self):
		#Code to really extract files

	def list(self):
		#Code to list contained files names