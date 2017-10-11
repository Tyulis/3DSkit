# -*- coding: utf-8 -*-
import os
import binascii
from .PyGBASM import Code
from .gen import generate, gensetup, disinventory
from plugins import *
from util.funcops import clearconsole, split

version = '1.0'

setups = {'basic': 'A basic setup'}

def setuplist():
	names = list(setups.keys())
	descs = [setups[name] for name in names]
	for i in range(0, len(names)):
		print('%d-%s: %s' % (i, names[i], descs[i]))
	print('')
	num = int(input('Your choice: '))
	clearconsole()
	name = names[num]
	code = readdata(name + '.s')
	c = Code(code)
	c.preprocess()
	c.assemble()
	bin = c.assembled[0]
	print(gensetup(bin))


def main(options):
	'''The plugin's entry point. Options are strings placed instead of the normal input name argument'''
	lang = options[0]
	opt = None
	while opt != 'q':
		clearconsole()
		print('ASMto8F v%s' % version)
		print('1-Inventory assembler')
		print('2-Inventory disassembler')
		print('3-Setup generator')
		print('	Q-Quit')
		print('')
		opt = input('Your choice: ').lower()
		clearconsole()
		if opt == 'q':
			continue
		elif opt == '1':
			print('1-Write code to compile')
			print('2-Compile code from a file')
			print('3-Generate inventory from hexadecimal')
			print('4-Generate simulated AR code')
			print('')
			o = input('Your choice: ')
			clearconsole()
			if o == '1':
				print('Enter a blank line to end')
				print('')
				code = ''
				ln = None
				while ln != '':
					ln = input('| ')
					code += ln + '\n'
				c = Code(code)
				c.preprocess()
				c.assemble()
				bin = c.assembled[0]
				print(generate(bin, lang))
			elif o == '2':
				bsdir = os.getcwd()
				filename = input('File Name: ')
				dir = os.path.dirname(filename)
				filename = filename.replace(dir, '')
				if filename.startswith(os.path.sep):
					filename.lstrip(os.path.sep)
				if dir != '':
					os.chdir(dir)
				code = read(filename)
				c = Code(code)
				c.preprocess()
				c.assemble()
				bin = c.assembled[0]
				os.chdir(bsdir)
				print(generate(bin, lang))
			elif o == '3':
				print('Write code as hexadecimal. Enter a blank line to end. Whitespaces are ignored')
				ln = None
				code = ''
				while ln != '':
					ln = input('$ ')
					code += ln
				code = code.replace(' ', '')
				bin = binascii.unhexlify(code).decode('ascii')
				print(generate(bin, lang))
			elif o == '4':
				print('Enter the AR (Gameshark) code you want to simulate')
				ar = input(': ')
				val = ar[2:4]
				addr = ar[6:8] + ar[4:6]
				code = readdata('ar.s') % (val, addr)
				c = Code(code)
				c.preprocess()
				c.assemble()
				bin = c.assembled[0]
				print(generate(bin, lang))
		elif opt == '2':
			print('1-Write inventory to disassemble')
			print('2-Disassemble inventory from file')
			print('3-Convert inventory to hexadecimal')
			print('4-Convert inventory to hexadecimal from file')
			print('')
			o = input('Your choice: ')
			clearconsole()
			if o == '1':
				print('Enter a blank line to end')
				print('Each line should be like that:')
				print('<item name> <item quantity>')
				print('')
				inv = ''
				ln = None
				while ln != '':
					ln = input('| ')
					inv += ln + '\n'
				code = disinventory(inv)
				c = Code.frombin(code)
				c.disassemble()
				print(c.code)
			elif o == '2':
				print('Each line should be like that:')
				print('<item name> <item quantity>')
				print('')
				filename = input('File name: ')
				inv = read(filename)
				code = disinventory(inv)
				c = Code.frombin(code)
				c.disassemble()
				print(c.code)
			elif o == '3':
				print('Enter a blank line to end')
				print('Each line should be like that:')
				print('<item name> <item quantity>')
				print('')
				inv = ''
				ln = None
				while ln != '':
					ln = input('| ')
					inv += ln + '\n'
				code = disinventory(inv)
				hx = binascii.hexlify(code).decode('ascii')
				l = split(hx, 2)
				print(' '.join(l))
			elif o == '4':
				filename = input('File name: ')
				inv = read(filename)
				code = disinventory(inv)
				hx = binascii.hexlify(code).decode('ascii')
				l = split(hx, 2)
				print(' '.join(l))
		elif opt == '3':
			print('1-Setups list')
			print('2-Write code to compile')
			print('3-Compile setup from file')
			print('')
			o = input('Your choice: ')
			clearconsole()
			if o == '1':
				setuplist()
			elif o == '2':
				print('Enter a blank line to end')
				print('')
				code = ''
				ln = None
				while ln != '':
					ln = input('| ')
					code += ln + '\n'
				c = Code(code)
				c.preprocess()
				c.assemble()
				bin = c.assembled[0]
				print(gensetup(bin, lang))
			elif o == '3':
				bsdir = os.getcwd()
				filename = input('File Name: ')
				dir = os.path.dirname(filename)
				filename = filename.replace(dir, '')
				if filename.startswith(os.path.sep):
					filename.lstrip(os.path.sep)
				if dir != '':
					os.chdir(dir)
				code = read(filename)
				c = Code(code)
				c.preprocess()
				c.assemble()
				bin = c.assembled[0]
				os.chdir(bsdir)
				print(gensetup(bin, lang))
		input('(return to menu)')
