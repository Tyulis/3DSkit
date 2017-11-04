#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import argparse
import pack
import unpack
import compress
import plugins
from util.fileops import bread, bwrite
from util.help import main_help
from util import error

__version__ = '1.17.34'


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


def pack_files(filenames, output, compression, format, isbigendian, opts):
	endian = '>' if isbigendian else '<'
	if format.upper() in pack.formats:
		print('Packing...')
		outnames = pack.pack(filenames, output, format, endian, opts)
		print('Packed!')
	else:
		error('Unknown format for repacking', 102)
	if compression is not None:
		print('Compressîng...')
		compress.compress(output, compression)
		print('Compressed')


def extract_files(filename, isbigendian, format, opts):
	endian = '>' if isbigendian else '<'
	content = bread(filename)
	format = unpack.recognize(content, filename, format)
	if format not in unpack.SKIP_DECOMPRESSION:
		compression = compress.recognize(content)
		if compression is not None:
			print('Compression: %s' % compression)
			print('Decompression...')
			content = compress.decompress(content, compression)
			print('Decompressed!')
		else:
			print('No compression')
	else:
		print('%s file: decompression skipped' % format)
	format = unpack.recognize(content, filename, format)
	if format is not None:
		print('%s file found' % format)
		print('Extracting...')
		unpack.extract(content, filename, format, endian, opts)
		print('Extracted!')
	else:
		if compression is not None:
			sname = list(filename.partition('.'))
			sname[0] += '_dec'
			filename = ''.join(sname)
			bwrite(content, filename)
			print('Wrote decompressed file to %s' % filename)
			error('Unrecognized format', 103)


def list_files(filename, isbigendian, format, opts):
	endian = '>' if isbigendian else '<'
	content = bread(filename)
	compression = compress.recognize(filename)
	if compression is not None:
		print('Decompression...')
		content = compress.decompress(content, compression)
		print('Decompressed!')
	else:
		print('No compression')
	format = unpack.recognize(content, filename, format)
	if format is not None:
		unpack.list_files(content, filename, format, endian, opts)
	else:
		error('Unrecognized format', 103)


def decompress_files(filename):
	content = bread(filename)
	compression = compress.recognize(content)
	if compression is None:
		error('The file is not compressed', 104)
	else:
		print('Compression: %s' % compression)
	print('Decompression...')
	content = compress.decompress(content, compression)
	sname = list(filename.partition('.'))
	sname[0] += '_dec'
	outname = ''.join(sname)
	bwrite(content, outname)
	print('Decompressed!')


def compress_file(filename, outname, compression):
	content = bread(filename)
	out = compress.compress(content, compression)
	bwrite(out, outname)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-H', '--detailed_help', help='Detailed help (you should read it the first time you use 3DSkit)', action='store_true')
	group = parser.add_mutually_exclusive_group()
	group.add_argument('-t', '--list', help='Lists the files contained in an archive', action='store_true')
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
	if args.detailed_help:
		main_help()
	elif args.list or args.extract:
		for filename in args.files:
			if args.list:
				list_files(filename, args.big, args.format, opts)
			elif args.extract:
				extract_files(filename, args.big, args.format, opts)
	elif args.pack:
		files = []
		basedir = os.getcwd()
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
		pack_files(files, args.out, args.compression, args.format.upper(), args.big, opts)
		os.chdir(basedir)
	elif args.decompress:
		for filename in args.files:
			decompress_files(filename)

	elif args.compress:
		if args.compression is None:
			error('You have to specify the compression type', 203)
		compress_file(args.files[0], args.out, args.compression)
	elif args.plugin is not None:
		plugins.run_plugin(args.plugin, args.files)
