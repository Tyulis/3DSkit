# -*- coding:utf-8 -*-
import os
import __main__
from util import error


def read(filename, encoding='utf-8'):
	try:
		file = open(filename, 'r', encoding=encoding)
	except OSError:
		error('File %s not found' % filename, 403)
	cnt = file.read()
	file.close()
	return cnt


def bread(filename):
	try:
		file = open(filename, 'rb')
	except OSError:
		error('File %s not found' % filename, 403)
	cnt = file.read()
	file.close()
	return cnt


def write(content, filename, encoding='utf-8'):
	file = open(filename, 'w', encoding=encoding)
	file.write(content)
	file.close()


def bwrite(content, filename):
	file = open(filename, 'wb')
	file.write(content)
	file.close()


def make_outdir(filename):
	outdir = os.path.splitext(filename)[0]
	if os.path.exists(outdir):
		if not os.path.isdir(outdir):
			outdir += '_'
	outdir += os.path.sep
	try:
		os.makedirs(outdir)
	except:
		pass
	return outdir


def make_outfile(filename, ext):
	outfile = os.path.splitext(filename)[0] + '.' + ext
	return outfile


def makedirs(filename):
	if os.path.splitext(filename)[0] != filename:
		dirname = os.path.dirname(filename)
	else:
		dirname = filename
	try:
		os.makedirs(dirname)
	except:
		pass
	return dirname + (os.path.sep if not dirname.endswith(os.path.sep) else '')


def basedir():
	os.chdir(__main__.basedir)


def path(*els):
	return os.path.join(*els)


def mkdir(dirname):
	try:
		os.mkdir(dirname)
	except FileExistsError:
		pass
