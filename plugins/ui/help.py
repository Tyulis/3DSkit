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
	h = '''3DSkit UI v0.1
		Use this to get a more user-friendly interface for 3DSkit
		To use it, you need Tkinter
		How to use:
			3DSkit.py -gui
	'''
	return h
