# -*- coding: utf-8 -*-

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
	h = '''newmod v1.0
		A plugin to create new 3DSkit modules
		How to use:
			Simply use -p newmod, and follow the program
	'''
	return h
