# -*- coding: utf-8 -*-

HELP = '''{{NAME}} v1.0
{{DESC}}
How to use:
	{{HOWTO}}
'''

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
	return HELP
