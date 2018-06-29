#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import sys
import time
import argparse
import pack
import unpack
import compress
import plugins
from io import BytesIO, StringIO
from util.help import main_help
from util import error

__version__ = '1.28.68'


def parse_opts(s):
	if s is None:
		return {}
	if len(s) == 0:
		return {}
	if s[0].startswith('{'):
		return parse_old_opts(s)
	opts = {}
	for opt in s:
		name, value = opt.split('=')
		opts[name] = value
	return opts
	

def parse_old_opts(s):
	opts = {}
	s = s.strip('()[]{}')
	ls = [el.strip() for el in s.split(',')]
	for opt in ls:
		name, value = opt.split('=')
		name = name.strip()
		value = value.strip()
		opts[name] = value
	return opts


def pack_files(filenames, output, compression, format, isbigendian, verbose, opts):
	endian = '>' if isbigendian else '<'
	for name in filenames:
		if not os.path.exists(name):
			error.FileNotFoundError('Input file %s is not found' % name)
	if format.upper() in pack.formats:
		print('Packing %s...' % output)
		pack.pack(filenames, output, format, endian, verbose, opts)
		print('Packed!')
	else:
		error.UnsupportedFormatError('3DSkit is currently unable to pack this format')
	if compression is not None:
		compress_file(output, compression, verbose, False)


def extract_files(filename, isbigendian, givenformat, verbose, opts):
	endian = '>' if isbigendian else '<'
	if os.path.isdir(filename):
		filenames = []
		for p, d, f in os.walk(filename):
			for name in f:
				filenames.append(os.path.join(p, name))
	else:
		filenames = [filename]
	for filename in filenames:
		print('\n--------%s--------' % filename)
		try:
			file = open(filename, 'rb')
		except FileNotFoundError:
			error.FileNotFoundError('File %s does not exist' % filename)
		format = unpack.recognize_file(file, givenformat)
		file.seek(0)
		if format is None:
			format = unpack.recognize_filename(filename, givenformat)
		if format not in unpack.SKIP_DECOMPRESSION:
			comp = compress.recognize(file)
			if comp == 0:
				if len(filenames) > 1:
					err = error.InvalidInputWarning
				else:
					err = error.InvalidInputError
				err("The given file is empty")
				continue
			if comp is not None:
				print('Compression: %s' % comp)
				print('Decompressing...')
				out = BytesIO()
				compress.decompress(file, out, comp, verbose)
				file.close()
				file = out
				print('Decompressed')
			else:
				print('No compression')
		else:
			print('No compression')
		format = unpack.recognize_file(file, givenformat)
		if format is None:
			format = unpack.recognize_filename(filename, givenformat)
		if format is None:  #still
			if len(filenames) > 1:
				err = error.UnrecognizedFormatWarning
			else:
				err = error.UnrecognizedFormatError
			err('Unrecognized format')
			continue
		print('%s file found' % format)
		print('Extracting...')
		unpack.extract(filename, file, format, endian, verbose, opts)
		print('Extracted')


def decompress_file(inname, outname, verbose):
	file = open(inname, 'rb')
	if outname is None:
		sname = list(inname.partition('.'))
		sname[0] += '_dec'
		outname = (''.join(sname)).replace('.cmp', '')
	out = open(outname, 'wb+')
	compression = compress.recognize(file)
	if compression is None:
		error.UnsupportedCompressionError('This file is not compressed, or 3DSkit currently does not support its compression')
	else:
		print('Compression: %s' % compression)
	print('Decompression...')
	compress.decompress(file, out, compression, verbose)
	file.close()
	print('Decompressed!')


def compress_file(inname, outname, compression, verbose):
	file = open(inname, 'rb')
	out = BytesIO()
	print('Compressing...')
	compress.compress(file, out, compression, verbose)
	file.close()
	if outname is None:
		outname = filename + '.cmp'
	out.seek(0)
	outfile = open(outname, 'wb')
	outfile.write(out.read())
	out.close()
	outfile.close()
	print('Compressed')


def main(args, opts):
	global basedir
	if args.quiet:
		initial_stdout = sys.stdout
		sys.stdout = StringIO()
	if args.debug:
		args.verbose = True
		error.debug = True
	if args.detailed_help:
		main_help()
	elif args.extract:
		for filename in args.files:
			extract_files(filename, args.big, args.format, args.verbose, opts)
	elif args.pack:
		files = []
		basedir = os.getcwd() + os.path.sep
		if args.format is None:
			error.ForgottenArgumentError('You have to specify the output format')
		if args.out is None:
			args.out = '%s.%s' % (os.path.splitext(args.files[0])[0], args.format.lower())
		if args.dir:
			try:
				os.chdir(args.files[0])
			except FileNotFoundError:
				error.FileNotFoundError('The given directory %s does not exist' % args.files[0])
			for path, dirs, filenames in os.walk(os.path.curdir):
				for filename in filenames:
					files.append(os.path.join(path, filename)[2:])  #strip the ./ or :\
		else:
			for file in args.files:
				if os.path.isdir(file):
					for path, dirs, filenames in os.walk(file):
						for filename in filenames:
							files.append(os.path.join(path, filename))
				else:
					files.append(file)
		pack_files(files, args.out, args.compression, args.format, args.big, args.verbose, opts)
		os.chdir(basedir)
	elif args.decompress:
		if len(args.files) > 1:
			for filename in args.files:
				decompress_file(filename, None, args.verbose)
		else:
			decompress_file(args.files[0], args.out, args.verbose)

	elif args.compress:
		if args.compression is None:
			error.ForgottenArgumentError('You have to specify the compression type')
		if len(args.files) > 1:
			for filename in args.files:
				compress_file(filename, None, args.compression, args.verbose)
		else:
			compress_file(args.files[0], args.out, args.compression, args.verbose)
	elif args.plugin is not None:
		plugins.run_plugin(args.plugin, args.files, args.verbose)
	else:
		if args.quiet:
			sys.stdout = initial_stdout
		return 1
	if args.quiet:
		sys.stdout = initial_stdout
	return 0


if __name__ == '__main__':
	try:
		parser = argparse.ArgumentParser()
		parser.add_argument('-H', '--detailed_help', help='Detailed help (you should read it the first time you use 3DSkit)', action='store_true')
		parser.add_argument('-v', '--verbose', help='Increases the verbosity of 3DSkit', action='store_true')
		parser.add_argument('-V', '--debug', help='Verbose mode, turns every 3DSkit error into Python exception (only useful for debugging)', action='store_true')
		parser.add_argument('-q', '--quiet', help='Run without any terminal output', action='store_true')
		group = parser.add_mutually_exclusive_group()
		group.add_argument('-x', '--extract', help='Extract files contained in the archive /  decompress the file if necessary and convert it to a readable format. On a directory, recursively extracts all contained files', action='store_true')
		group.add_argument('-p', '--pack', help='Pack files into an archive, or convert it to a console format', action='store_true')
		group.add_argument('-D', '--decompress', help='Decompress the input files', action='store_true')
		group.add_argument('-C', '--compress', help='Compress the input file', action='store_true')
		group.add_argument('-g', '--plugin', help='Run a plugin')
		parser.add_argument('-d', '--dir', help='Use a directory to pack an archive. The root will be the directory. Very recommended.', action='store_true')
		parser.add_argument('-f', '--format', help='Format to repack, or input format (to extract. See the formats section of the help for more infos)')
		group = parser.add_mutually_exclusive_group()
		group.add_argument('-l', '--little', help='Little endian (for 3DS / NDS files)', action='store_true')
		group.add_argument('-b', '--big', help='Big endian (for WiiU files)', action='store_true')
		parser.add_argument('-o', '--out', help='Output file name (only for repacking). If not given, the output file name will be automatically determined')
		parser.add_argument('-c', '--compression', help='Output compression type')
		parser.add_argument('-O', '--options', help='Format-specific options, see help for details', action='append')
		parser.add_argument('files', help='Name of the file to convert or to pack into an archive', nargs='*')
		args = parser.parse_args()
		opts = parse_opts(args.options)
		if args.debug:
			starttime = time.time()
		result = main(args, opts)
		if args.debug:
			endtime = time.time()
			print('Actions took %.3f s' % (endtime - starttime))
		if result == 1:
			parser.print_help()
	except KeyboardInterrupt:
		error.UserInterrupt("User interruption")
