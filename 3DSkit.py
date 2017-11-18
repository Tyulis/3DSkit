#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import argparse
import pack
import unpack
import compress
import plugins
from io import BytesIO
from util.help import main_help
from util import error

__version__ = '1.19.44'


def parse_opts(s):
	if s is None:
		return {}
	opts = {}
	s = s.strip('()[]{}')
	ls = [el.strip() for el in s.split(';')]
	for opt in ls:
		name, value = opt.split('=')
		name = name.strip()
		value = value.strip()
		opts[name] = value
	return opts


def pack_files(filenames, output, compression, format, isbigendian, verbose, opts):
	endian = '>' if isbigendian else '<'
	if format.upper() in pack.formats:
		print('Packing...')
		pack.pack(filenames, output, format, endian, verbose, opts)
		print('Packed!')
	else:
		error('Unknown format for repacking', 102)
	if compression is not None:
		compress_file(output, compression, verbose, False)


def extract_files(filename, isbigendian, format, verbose, opts):
	endian = '>' if isbigendian else '<'
	try:
		file = open(filename, 'rb')
	except FileNotFoundError:
		error('File %s does not exist' % filename, 404)
	format = unpack.recognize(filename, format)
	if format not in unpack.SKIP_DECOMPRESSION:
		comp = compress.recognize(file)
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
	if format is None:
		format = unpack.recognize(file, format)
		if format is None:
			error('Unrecognized format', 103)
	print('%s file found' % format)
	print('Extracting...')
	unpack.extract(filename, file, format, endian, verbose, opts)
	print('Extracted')


def decompress_file(filename, verbose):
	file = open(filename, 'rb')
	sname = list(filename.partition('.'))
	sname[0] += '_dec'
	filename = ''.join(sname)
	out = open(filename.replace('.cmp', ''), 'wb+')
	compression = compress.recognize(file)
	if compression is None:
		error('The file is not compressed', 104)
	else:
		print('Compression: %s' % compression)
	print('Decompression...')
	compress.decompress(file, out, compression, verbose)
	file.close()
	print('Decompressed!')


def compress_file(filename, compression, verbose, separate=True):
	file = open(filename, 'rb')
	out = BytesIO()
	print('Compressing...')
	compress.compress(file, out, compression, verbose)
	file.close()
	if separate:
		outname = filename + '.cmp'
	else:
		outname = filename
	out.seek(0)
	outfile = open(outname, 'wb')
	outfile.write(out.read())
	out.close()
	outfile.close()
	print('Compressed')


def main(args, opts, parser=None):
	global basedir
	if args.detailed_help:
		main_help()
	elif args.extract:
		for filename in args.files:
			extract_files(filename, args.big, args.format, args.verbose, opts)
	elif args.pack:
		files = []
		basedir = os.getcwd() + os.path.sep
		if args.out is None:
			error('You have to specify the output name', 201)
		if args.format is None:
			error('You have to specify the output format', 202)
		if args.dir:
			os.chdir(args.files[0])
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
		for filename in args.files:
			decompress_file(filename, args.verbose)

	elif args.compress:
		if args.compression is None:
			error('You have to specify the compression type', 203)
		for filename in args.files:
			compress_file(filename, args.compression, args.verbose)
	elif args.plugin is not None:
		plugins.run_plugin(args.plugin, args.files, args.verbose)
	else:
		if parser is not None:
			parser.print_help()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-H', '--detailed_help', help='Detailed help (you should read it the first time you use 3DSkit)', action='store_true')
	parser.add_argument('-v', '--verbose', help='Increases the verbosity of 3DSkit', action='store_true')
	group = parser.add_mutually_exclusive_group()
	group.add_argument('-x', '--extract', help='Extract files contained in the archive /  decompress the file if necessary and convert it to a readable format', action='store_true')
	group.add_argument('-p', '--pack', help='Pack files into an archive, or convert it to a console format', action='store_true')
	group.add_argument('-D', '--decompress', help='Decompress the input files', action='store_true')
	group.add_argument('-C', '--compress', help='Compress the input file', action='store_true')
	group.add_argument('-g', '--plugin', help='Run a plugin')
	parser.add_argument('-d', '--dir', help='Use a directory to pack an archive. The root will be the directory. Very recommended.', action='store_true')
	parser.add_argument('-f', '--format', help='Format to repack, or input format (to extract. See the formats section of the help for more infos)')
	group = parser.add_mutually_exclusive_group()
	group.add_argument('-l', '--little', help='Little endian (for 3DS / NDS files)', action='store_true')
	group.add_argument('-b', '--big', help='Big endian (for WiiU files)', action='store_true')
	parser.add_argument('-o', '--out', help='Output file name (only for repacking)')
	parser.add_argument('-c', '--compression', help='Output compression type')
	parser.add_argument('-O', '--options', help='Format-specific options, see help for details')
	parser.add_argument('files', help='Name of the file to convert or to pack into an archive', nargs='*')
	args = parser.parse_args()
	opts = parse_opts(args.options)
	main(args, opts, parser)
