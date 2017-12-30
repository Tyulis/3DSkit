# -*- coding:utf-8 -*-
from util.filesystem import read
from util.utils import clearconsole
import os


def menu():
	clearconsole()
	print('3DSkit detailed help')
	print('====================')
	print('')
	print('Select one of the following sections:')
	print('1-Informations on supported formats')
	print('2-Informations on output formats')
	print('3-Detailed help on program\'s options')
	print('4-Plugins help')
	print('5-Guide to create new modules and plugins')
	print('6-Documentation on the 3DSkit\'s developper interface')
	print('    Enter Q to quit')
	print('')
	opt = input('Your choice: ').lower()
	return opt


def plugins_help():
	plugins = []
	for el in os.scandir('plugins'):
		if el.is_dir():
			plugins.append(el.name)
	plugins.sort()
	helps = {}
	refs = {i: name for i, name in enumerate(plugins)}
	menu = ['%d-%s' % (i, name) for i, name in enumerate(plugins)]
	menu.append('%d-Return to menu' % len(plugins))
	for i, name in enumerate(plugins):
		mod = __import__('plugins.%s.help' % name)
		exec('helps[name] = mod.%s.help.make_help()' % name, globals(), locals())
	opt = -1
	menu = '\n'.join(menu)
	while opt != len(plugins):
		clearconsole()
		print(menu)
		opt = int(input('Your choice: '))
		clearconsole()
		if opt == len(plugins):
			break
		cnt = helps[refs[opt]].split(';;;;')
		for part in cnt:
			print(part)
			input('----')
		input('(return to menu)')


def main_help():
	opt = ''
	while opt != 'q':
		opt = menu()
		clearconsole()
		if opt == '1':
			cnt = read(os.path.join('util', 'data', 'formats_tbl.txt')).split(';;;;')
		elif opt == '2':
			cnt = read(os.path.join('util', 'data', 'output.txt')).split(';;;;')
		elif opt == '3':
			cnt = read(os.path.join('util', 'data', 'options.txt')).split(';;;;')
		elif opt == '4':
			plugins_help()
			continue
		elif opt == '5':
			cnt = read(os.path.join('util', 'data', 'contrib.txt')).split(';;;;')
		elif opt == '6':
			cnt = read(os.path.join('util', 'data', 'dev.txt')).split(';;;;')
		else:
			continue
		for block in cnt:
			print(block)
			input('----')
		input('(Return to menu)')
