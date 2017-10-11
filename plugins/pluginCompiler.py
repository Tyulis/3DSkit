# -*- coding:utf-8 -*-
import os
import sys
import shutil
import argparse

HELP_TEMPLATE = """# -*- coding: utf-8 -*-

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
	h = '''MyPlugin v1.0
		A 3DSkit plugin which is used as an example
		How to use:
			Simply use -p MyPlugin
	'''
	return h
"""

MAIN_TEMPLATE = """# -*- coding: utf-8 -*-

def main(options):
	'''The plugin's entry point. Options are strings placed instead of the normal input name argument'''
	print('Hello 3DSkit!')
"""

def read(filename):
	file = open(filename, 'r', encoding='utf-8')
	cnt = file.read()
	file.close()
	return cnt

def write(content, filename):
	file = open(filename, 'w', encoding='utf-8')
	file.write(content)
	file.close()

def compile_plugin(name):
	ls = os.listdir(name)
	if ('help.py' in ls) and ('main.py' in ls):
		print('OK!')
	else:
		print('help.py or main.py are missing from the root directory of the plugin')
		sys.exit(1)

def new_plugin(name):
	try: os.mkdir(name)
	except: pass
	try: os.mkdir(os.path.join(name, 'data'))
	except: pass
	if not name.endswith(os.path.sep):
		dirname = name + os.path.sep
	else:
		dirname = name
	write(HELP_TEMPLATE, dirname + 'help.py')
	write(MAIN_TEMPLATE, dirname + 'main.py')
	write('', dirname + '__init__.py')

if __name__ == '__main__':
	p = argparse.ArgumentParser()
	group = p.add_mutually_exclusive_group()
	group.add_argument('-n', '--new', help='Creates a new plugin project')
	group.add_argument('-c', '--compile', help='Compile a plugin project')
	args = p.parse_args()
	if args.new is not None:
		new_plugin(args.new)
	elif args.compile is not None:
		compile_plugin(args.compile)
