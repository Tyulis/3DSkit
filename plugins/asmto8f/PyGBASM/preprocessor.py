# -*- coding:utf-8 -*-
from ._intern import *
from .assembler import getsize, NUM_OPS


def preprocess(scode):
	rominfo = {}
	bcode = [lsplit(ln, ';')[0].strip().lower() for ln in scode.splitlines()]
	bcode = [ln for ln in bcode if ln != '']
	lcode = []
	for ln in bcode:
		if ln.startswith('.incbin'):
			lcode.append(ln)
		elif ln.startswith('.inc'):
			name = ln.split()[-1]
			c = read(name)
			c = [lsplit(ln, ';')[0].strip().lower() for ln in c.splitlines()]
			c = [ln for ln in c if ln != '']
			lcode += c
		else:
			lcode.append(ln)
	code = []
	for ln in bcode:
		if ln.startswith('.') and not ln.startswith(('.def', '.inc', '.incbin', '.db', '.org')):
			l = lsplit(ln)
			rominfo[l[0].strip('. ')] = l[1:]
		else:
			inter = ln
			ln = ln.replace('  ', ' ')
			while ln != inter:
				inter = ln
				ln = ln.replace('  ', ' ')
			code.append(ln)
	defs = {}
	icode = code
	code = []
	for ln in icode:
		if ln.startswith('.def'):
			l = ln.split()
			defs[l[1]] = l[2]
		else:
			code.append(ln)
	for i, ln in enumerate(code):
		l = lsplit(ln)
		op = l[0]
		if op in defs.keys():
			op = defs[op]
		try:
			args = [el.strip() for el in l[1].split(',')]
			for j, arg in enumerate(args):
				for df in defs.keys():
					if df in arg and not arg.startswith('"'):
						args[j] = arg.replace(df, defs[df])
		except IndexError:
			args = []
		args = ','.join(args)
		if args != '':
			ln = op + ' ' + args
		else:
			ln = op
		code[i] = ln
	sections = {0: []}
	actseq = sections[0]
	for ln in code:
		if ln.startswith('.org'):
			org = toint(lsplit(ln)[1])
			sections[org] = []
			actseq = sections[org]
		elif ln.startswith(NUM_OPS):
			if not lsplit(ln)[1].startswith(('a,', 'hl,')):
				l = lsplit(ln)
				actseq.append(l[0] + ' a,' + l[1])
			else:
				actseq.append(ln)
		elif ln.startswith(('jr', 'jp', 'call')):
			if ',' not in lsplit(ln)[1]:
				l = lsplit(ln)
				actseq.append(l[0] + ' nn,' + l[1])
			else:
				actseq.append(ln)
		else:
			actseq.append(ln)
	labels = {}
	for offset in sections.keys():
		sec = sections[offset]
		for i, ln in enumerate(sec):
			if ln.endswith(':'):
				label = ln.strip(':')
				val = getsize(sec[:i])
				labels[label] = offset + val
	return rominfo, sections, labels
