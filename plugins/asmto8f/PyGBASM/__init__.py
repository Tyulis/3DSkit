# -*- coding:utf-8 -*-
from .preprocessor import preprocess
from .assembler import assemble
from .makerom import makerom, disrom
from .disassembler import disassemble

class Code (object):
	def __init__(self, code):
		self.code = code
	
	def preprocess(self):
		self.rominfo, self.sections, self.labels = preprocess(self.code)
	
	def assemble(self):
		self.assembled = assemble(self.sections, self.labels)
	
	def makerom(self, outname):
		self.rom = makerom(self.assembled, self.rominfo, outname)
	
	@classmethod
	def frombin(cls, binary):
		ins = object.__new__(cls)
		ins.assembled = {0: binary}
		return ins
	
	@classmethod
	def fromrom(cls, romname):
		ins = object.__new__(cls)
		ins.assembled = disrom(romname)
		return ins
	
	def disassemble(self):
		self.code = disassemble(self.assembled)
