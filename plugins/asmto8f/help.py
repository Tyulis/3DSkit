# -*- coding: utf-8 -*-
from .main import version

def make_help():
	'''Must return the plugin's help as a string.
	To split into parts like this:
		A first part
		----
		A second part
	Continuable by pressing enter, place 4 semicolons (;;;;) like that:
		A first part
		;;;;
		A second part
	'''
	h = '''ASMto8F v%s
		A 3DSkit plugin used as a test for the plugin system. Allow to convert assembler code into Pokemon R/B 8F code execution setup and items
		How to use:
			python3 3DSkit.py -g asmto8f [fr | us]
		fr / us : to use with FR or US version of the game
		Then follow instructions
	''' % version
	return h
