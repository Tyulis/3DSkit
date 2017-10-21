3DSkit
======

3DSkit is a tool to extract and repack many files formats found on the NDS, DSi, 3DS and WiiU
For help and informations on use and options, read the wiki or use the -H option of 3DSkit

What does 3DSkit?
=================

3DSkit can:

*	Extract and convert many files formats found in 3DS, WiiU, and why not NDS games
*	Pack or repack them
*	Decompress, and soon compress these files with their original compression

How to use
==========

	usage: 3DSkit.py [-h] [-H] [-t | -x | -p | -D | -C | -g PLUGIN] [-d] [-f FORMAT] [-l | -b] [-o OUT] [-c COMPRESSION] [-O OPTIONS] [files [files ...]]

	positional arguments:
		files                 Name of file to convert or to pack into an archive

	optional arguments:
		-h, --help            show this help message and exit
		-H, --detailed_help   Detailed help (you should read it the first time you use 3DSkit)
		-t, --list            Lists the files contained in an archive
		-x, --extract         Extract files contained in the archive / decompress the file if necessary and convert it to a readable format
		-p, --pack            Pack files into an archive, or convert it to a console format
		-D, --decompress      Decompress the input files
		-C, --compress        Compress the input file
		-g PLUGIN, --plugin PLUGIN
								Run a plugin
		-d, --dir             Use a directory to pack an archive. The root will be the directory. Very recommended.
		-f FORMAT, --format FORMAT
								Format to repack, or input format (to extract. See the formats section of the help for more infos)
		-l, --little          Little endian (for 3DS / NDS files)
		-b, --big             Big endian (for WiiU files)
		-o OUT, --out OUT     Output file name (only for repacking)
		-c COMPRESSION, --compression COMPRESSION
								Output compression type
		-O OPTIONS, --options OPTIONS
								Format-specific options, see help for details

Dependencies
============
To access all 3DSkit functionnalities, you need:

*	python3 (tested under python 3.5 and 3.6, but should work with previous versions)
*	Pillow (Fork of PIL, use sudo apt-get install python3-pil)
*	struct (it should be installed by default with python)

Contributing
============

To submit an issue, don't forget to check i the issue has not been already reported, and fixed (check also in te closed).
Check also if you have the last commit
Then, if you really found a new issue, you should precise:

*	Your python version
*	The error output in the console
*	Attach the concerned file and if possible and necessary, a screenshot of the problem in the game.

Pull requests are grantly appreciated.
