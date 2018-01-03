3DSkit
======

3DSkit is a tool to extract and repack many files formats found on the NDS, DSi, 3DS and WiiU
For help and informations on use and options, read the wiki or use the -H option of 3DSkit

What does 3DSkit?
=================

3DSkit can:

*	Extract and convert many files formats found in 3DS, WiiU, and NDS games
*	Pack or repack them
*	Decompress and compress these files from and to their original compression

How to use
==========

	usage: 3DSkit.py [-h] [-H] [-v] [-V] [-q] [-x | -p | -D | -C | -g PLUGIN] [-d]
                 [-f FORMAT] [-l | -b] [-o OUT] [-c COMPRESSION] [-O OPTIONS]
                 [files [files ...]]

	positional arguments:
	  files                 Name of the file(s) to convert or to pack into an archive
	
	optional arguments:
	  -h, --help            show this help message and exit
	  -H, --detailed_help   Detailed help (you should read it the first time you
	                        use 3DSkit)
	  -v, --verbose         Increases the verbosity of 3DSkit
	  -V, --debug           Verbose mode, turns every 3DSkit error into Python
	                        exception (only useful for debugging)
	  -q, --quiet           Run without any terminal output
	  -x, --extract         Extract files contained in the archive / decompress
	                        the file if necessary and convert it to a readable
	                        format. On a directory, recursively extracts all
	                        contained files
	  -p, --pack            Pack files into an archive, or convert it to a console
	                        format
	  -D, --decompress      Decompress the input files
	  -C, --compress        Compress the input file
	  -g PLUGIN, --plugin PLUGIN
	                        Run a plugin
	  -d, --dir             Use a directory to pack an archive. The root will be
	                        the directory. Very recommended.
	  -f FORMAT, --format FORMAT
	                        Format to repack, or input format (to extract. See the
	                        formats section of the help for more infos)
	  -l, --little          Little endian (for 3DS / NDS files)
	  -b, --big             Big endian (for WiiU files)
	  -o OUT, --out OUT     Output file name (only for repacking). If not given,
	                        the output file name will be automatically determined
	  -c COMPRESSION, --compression COMPRESSION
	                        Output compression type
	  -O OPTIONS, --options OPTIONS
	                        Format-specific options, see help for details

Examples
--------

Extract a DARC archive:

	python3 3DSkit.py -x archive.darc

Extract a 3DS ROM (NCCH) without doing hash checks

	python3 3DSkit.py -x -O{dochecks=false} my_ncch_partition.cxi

Convert a BFLIM image in verbose mode:

	python3 3DSkit.py -xv myTexture.bflim

Convert into a BFLYT file with a custom output name:

	python3 3DSkit.py -pf BFLYT -o myLayout.bflyt a-layout.tflyt

Pack a folder into a version 6 GARC archive in verbose mode

	python3 3DSkit.py -pdvf GARC -O{version=6} any-directory

Pack three files into an SARC file and compress it in LZ11

	python3 3DSkit.py -pf SARC -c LZ11 file1 file2 file3

Compress a DARC file without any console output with a specific output name

	python3 3DSkit.py -Cqc LZ11 -o archive_LZ.darc archive.darc

Decompress the previous DARC file

	python3 3DSkit.py -D archive_LZ.darc

Format-Specific options
=======================

Specify these options with the -O option (see options help)
You can specify them with -O(option=value) or -O{option1=value1,option2=value2}

**At extraction**:

*	NCCH/ExeFS:
	*	**dochecks**: If "false", don't checks if the contents hashes match. Defaults to true.
	*	**dumpfs**: If "true", dumps the ExeFS and the RomFS images as exefs.bin and romfs.bin before extracting them. Defaults to false

**At packing**:

*	BFLIM:
	*	**format**: Specify the color format. Can be: RGBA8, RGBA4, RGB8, RGB565, RGBA5551, LA8, LA4, L8, L4, A8, A4. Note that this will have an effect on color quality and file size, but not on the functionment of the game, you can repack in a different format from original without any problem, for example for ETC1, not supported for packing. Defaults to RGBA8
		
	*	**swizzle**: Specify the texture swizzling (see console output at extraction). Can be 0 (none), 4 (90º rotation) or 8 (transposition). Defaults to 0
*	GARC:
	*	**version**: Specify the version of the output file. Can be 4 or 6, look at the console output during the extraction. Defaults to 6.
*	mini:
	*	**type**: Specify the file type (2 chars magic number) to use for the output file. Look at the console output during the extraction. If you don't use the right magic, it can prevent the game to load the packed file. Defaults to BL.

Supported formats
=================

Info on the table:

*	X (Extract): 3DSkit can extract these files
*	P (Pack): 3DSkit can create and pack these files from scratch
*	R (Repack): 3DSkit can only repack files which have been extracted by 3DSkit
*	Extensions: Frequent file extensions or names for this format (all formats can sometimes have the .bin extension)

Crosses:

*	x : full support
*	e : experimental / untested
*	p : partial support
*	n : not functional
*	b : bugged
*	  : no support
	
Output: Output format at extracting. See the output formats help for informations

	Format | X | P | R | Extensions          | Output
	--------------------------------------------------
	ALYT   | x |   | x | .alyt               | files
	BCSAR  | b |   |   | .bcsar              | files
	BCSTM  | x |   |   | .bcstm              | WAV
	BFLAN  | p |   |   | .bflan              | TX
	BFLIM  | x | x |   | .bflim              | PNG
	BFLYT  | x |   | x | .bflyt              | TX
	CBMD   | e |   |   | .bnr banner[.bin]   | files
	DARC   | x | e |   | .arc .bcma (...)    | files
	GARC   | x |   | x | .garc (none)        | files
	GFA    | x |   |   | .gfa                | files
	mini   | x |   | x | .mini .bl .wd (none)| files
	NARC   | p |   |   | .narc               | files
	NCCH   | x |   |   | .app .cxi .cfa      | sections
	NDS    | x |   |   | .nds                | files
	SARC   | x | x |   | .sarc .arc .szs     | files

There is also:

*	unpack.ExtHeader:
	*	Unpacks NCCH extended headers (automatic when unpacking an NCCH partition). Needs a specific file name (exheader.bin, extheader.bin or DecryptedExtHeader.bin) to be recognized. Outputs as a TXTree
*	unpack.ExeFS:
	*	Unpack ExeFS images (automatic when unpacking an NCCH partition). The file must have the .exefs extension or be "exefs.bin" or "DecryptedExeFS.bin" to be recognized
*	unpack.RomFS:
	*	Unpack RomFS images (automatic when unpacking an NCCH partition). The file must have the .romfs extension or be "romfs.bin" or "DecryptedRomFS.bin" to be recognized

Compressions:

D: Decompressible

C: Compressible

	Compression | D | C | Extensions
	-------------------------------------------------
	LZ10        | x | x | (none) .cmp .l *_LZ.bin .LZ
	LZ11        | x | x | (none) .cmp .l *_LZ.bin .LZ
	LZH8        | x |   | .cx
	Yaz0        | x |   | .szs
	ETC1        | x |   | Texture compression in BFLIM

Dependencies
============
To access all 3DSkit functionnalities, you need:

*	Python 3.5+
*	struct (it should be installed by default with python3)
*	Pillow (Fork of PIL, use sudo apt-get install python3-pil or pip3 install pillow)
*	Numpy

Note that all the non-built-in dependencies are only needed by some modules -- if you don't have them, you can still use the others

Contributing
============

Before submitting an issue, take a look at the program's help (-H option), your problem can be normal, or already explained there.
Don't forget to check if the issue has not been already reported, or fixed (check also in the closed ones).
Check also if you have the last commit
Then, if you really found a new issue, you should precise:

*	Your python version
*	The console output (in verbose mode, -v option)
*	Attach the concerned file and if possible and necessary, a screenshot of the problem in the game.

Pull requests are gladly appreciated. You can see a guide for the 3DSkit's developer interface in the program's help (-H option), and a documentation for rawutil in the [rawutil's repo](https://github.com/Tyulis/rawutil)
