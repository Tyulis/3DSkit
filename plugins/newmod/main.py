# -*- coding: utf-8 -*-
import os
from plugins import *
from util.funcops import clearconsole
from util.fileops import *

def menu():
	print('1-Format unpacker')
	print('2-Format packer')
	print('3-Plugin')
	print('\tQ-Quit')
	print('')
	opt = input('Your choice: ')
	return opt

def textinput(prompt):
	print(prompt)
	print('Enter 2 blank lines to end writing')
	blanks = 0
	final = ''
	while blanks < 2:
		line = input('| ')
		if line == '':
			blanks += 1
		else:
			blanks = 0
		final += line + '\n'
	clearconsole()
	return final.rstrip()

def create_unpacker():
	template = readdata('unpack.py')
	format = input('Format name: ')
	outext = input('Output file name extension: ')
	outname = path('unpack', '%s.py' % format)
	template = template.replace('{{FORMAT}}', format).replace('{{OUTEXT}}', outext)
	archive = input('Is your format an archive format (y/n)? ').lower() in 'yoj'
	out = []
	for ln in template.splitlines():
		if ln.startswith('>a'):
			if archive:
				out.append(ln.replace('>a', ''))
		elif ln.startswith('>f'):
			if not archive:
				out.append(ln.replace('>f', ''))
		else:
			out.append(ln)
	write('\n'.join(out), outname)

def create_packer():
	template = readdata('pack.py')
	format = input('Format name: ')
	template = template.replace('{{FORMAT}}', format)
	outname = path('pack', '%s.py' % format)
	archive = input('Is your format an archive format (y/n)? ').lower() in 'yoj'
	out = []
	for ln in template.splitlines():
		if ln.startswith('>a'):
			if archive:
				out.append(ln.replace('>a', ''))
		elif ln.startswith('>f'):
			if not archive:
				out.append(ln.replace('>f', ''))
		else:
			out.append(ln)
	write('\n'.join(out), outname)

def create_plugin():
	name = input("Plugin's name: ")
	outpath = path('plugins', name, '')
	mkdir(outpath)
	mkdir(outpath + 'data')
	helpname = outpath + 'help.py'
	mainname = outpath + 'main.py'
	initname = outpath + '__init__.py'
	description = textinput('Enter the plugin description')
	howto = textinput('Enter the usage help')
	help_temp = readdata('help.py')
	help = help_temp.replace('{{NAME}}', name)
	help = help.replace('{{DESC}}', description)
	help = help.replace('{{HOWTO}}', howto)
	write(help, helpname)
	main_temp = readdata('main.py')
	write(main_temp, mainname)
	write('', initname)
	clearconsole()

def main(options, verbose):
	'''The plugin's entry point. Options are strings placed instead of the normal input name argument'''
	opt = menu().lower()
	clearconsole()
	if opt == 'q':
		quit()
	elif opt == '1':
		create_unpacker()
	elif opt == '2':
		create_packer()
	elif opt == '3':
		create_plugin()
	clearconsole()
