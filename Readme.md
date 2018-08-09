3DSkit
======

3DSkit is a tool to extract and repack many files formats found on the NDS, DSi, 3DS, WiiU and Switch.

What does 3DSkit?
=================

3DSkit can:

*	Extract and convert many files formats found in NDS, 3DS, WiiU, Switch games
*	Pack or repack them
*	Decompress and compress these files from and to their original compression

Preparing
=========

Now, some 3DSkit modules are based on libkit. This is a "virtual" module, which can be either
c3DSkit, a C extension, or py3DSkit its equivalent in pure Python.
This means that you can use 3DSkit directly as is, without any setup : 
all 3DSkit modules can work without c3DSkit.

However, some formats are VERY, very slow to process in pure Python : 
for example, textures, fonts, audio... So if you can, build c3DSkit, which
computes them much faster (sometimes 10000x or more).

To install c3DSkit, just come into the c3DSkit directory and run `python3 setup.py install`.
You need a working C compiler and the Python includes (should be included in your python installation)
When you update it, it may be required to remove the `build/` directory.
If you don't have the admin rights, try `python3 setup.py install --user`.

How to use
==========

	usage: 3DSkit.py [-h] [-v] [-V] [-q] [-x | -p | -D | -C | -g PLUGIN] [-d]
				[-f FORMAT] [-l | -b] [-o OUT] [-c COMPRESSION]
				[--ckit | --pykit] [-O OPTIONS]
				[files [files ...]]

	Positional arguments:
	files                 Name of the file to convert or to pack into an archive

	Optional arguments:
	-h, --help            show this help message and exit
	-v, --verbose         Increases the verbosity of 3DSkit
	-V, --debug           Debugging mode, turns every 3DSkit error into Python
	                      exceptions (only useful for debugging)
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
	-l, --little          Little endian (for 3DS / NDS files), default
	-b, --big             Big endian (for WiiU files)
	-o OUT, --out OUT     Output file name (only for repacking). If not given,
	                      the output file name will be automatically determined
	-c COMPRESSION, --compression COMPRESSION
	                      Output compression type
	--ckit                Force use of c3DSkit as libkit
	--pykit               Force use of py3DSkit as libkit
	-O OPTIONS, --options OPTIONS
	                      Format-specific options, see below for details


Examples
--------

Extract a DARC archive:

	python3 3DSkit.py -x archive.darc
	
Recursively extract all the contents of a directory:

	python3 3DSkit.py -x my-directory/

Extract a 3DS ROM (NCCH) without doing hash checks

	python3 3DSkit.py -x -O dochecks=false my_ncch_partition.cxi

Convert a BFLIM image in verbose mode:

	python3 3DSkit.py -xv myTexture.bflim

Convert into a BFLYT file with a custom output name:

	python3 3DSkit.py -pf BFLYT -o myLayout.bflyt a-layout.tflyt

Pack a folder into a version 6 GARC archive in verbose mode

	python3 3DSkit.py -pdvf GARC -O version=6 any-directory

Pack three files into an SARC file and compress it in LZ11

	python3 3DSkit.py -pf SARC -c LZ11 file1 file2 file3

Pack a BFLIM texture for the WiiU (big endian):

	python3 3DSkit.py -pbf BFLIM my_texture.png

Pack two BCSTM respectively with one and two tracks, the second with a loop.

	python3 3DSkit.py -pf BCSTM -o oneTrackTest.bcstm track1.wav
	python3 3DSkit.py -pf BCSTM -o twoTracksExample.bcstm -O loop=100000-2100000 track1.wav track2.wav

Compress a DARC file without any console output and with a specific output file name

	python3 3DSkit.py -Cqc LZ11 -o archive_LZ.darc archive.darc

Decompress the previous DARC file

	python3 3DSkit.py -D archive_LZ.darc

Format-Specific options
=======================

Specify these options with the -O option
You can specify them with `-O option1=value1 -O option2=value2 ...`

**At extraction**:

*	NCCH/ExeFS:
	*	**dochecks**: If `false`, don't checks if the contents hashes match. If you get a `HashMismatchError`, it's theorically right, so change this only if you're sure of the integrity of your file. Defaults to `true`.
	*	**dumpfs**: If `true`, dumps the ExeFS and the RomFS images as exefs.bin and romfs.bin before extracting them. Defaults to `false`
*	GARC:
	*	**skipdec**: If `true`, force the module to not decompress the contained files. This may be useful if it detects a compression while there is not, but only in that case. Defauts to `false`
*	BFFNT:
	*	**origin**: Sets the original console from which the file comes. Can be set to `CTR` for 3DS, `CAFE` for WiiU or `NX` for Switch. Try this if you get errors or glitched output. If not specified, tries to auto-detect from the file's version.
*	MSBT:
	*	**showescapes**: If `false`, just erases not displayed characters, else show them as `\\uXXXX`. Defaults to `true`

**At packing**:

*	BFLIM:
	*	**format**: Specify the color format. Can be: RGBA8, RGBA4, RGB8, RGB565, RGBA5551, LA8, LA4, L8, L4, A8, A4. Note that this will have an effect on color quality and file size, but not on the functionment of the game, you can repack in a different format from original without any problem, for example for ETC1, not supported for packing. Defaults to `RGBA8`
	*	**swizzle**: Specify the texture swizzling (see console output at extraction). Can be 0 (none), 4 (90º rotation) or 8 (transposition). Defaults to `0`.
*	GARC:
	*	**version**: Specify the version of the output file. Can be 4 or 6, look at the console output during the extraction. Defaults to `6`.
*	mini:
	*	**type**: Specify the file type (2 chars magic number) to use for the output file. Look at the console output during the extraction. If you don't use the right magic, it can prevent the game to load the packed file. Defaults to `BL`.
*	BCSTM:
	*	**format**: Specify the audio format. Can be `DSP-ADPCM`, `IMA-ADPCM`, `PCM16` or `PCM8` (currently only DSP-ADPCM is supported). Defaults to `DSP-ADPCM`
	*	**loop**: Makes a looping BCSTM. Must be of the form <start>-<end> (eg. 688123-2976543). If not given, the packed BCSTM won't loop at all.
	*	**writetracks**: If a BCSTM contains only 1 track, sometimes the track is explicitely written in the file, and sometimes not. If you want to pack only one track, you can set it to `false` to not write the track info. Defaults to true, change only if default don't work. _Note that a track is theorically a standalone stream, it's different than a channel_

Supported formats
=================

Info on the table:

*	X (Extract): 3DSkit can extract these files
*	P (Pack): 3DSkit can create and pack these files from scratch
*	R (Repack): 3DSkit can only repack files which have been extracted by 3DSkit
*	Extensions: Frequent file extensions or names for this format (all formats can sometimes have the .bin extension)

Crosses:

*	x : full support
*	e : experimental
*	p : partial support
*	  : no support
	
Output: Output format at extracting. See the output formats help for informations
As explained previously, modules which use c3DSkit may be very slow in pure Python and much faster if you have c3DSkit installed

	Format | X | P | R | Extensions          | Output  | Uses libkit
	-----------------------------------------------------------------
	ALYT   | x |   | x | .alyt               | files   | 
	BCGRP  | x |   |   | .bcgrp              | files   |
	BCSAR  | e |   |   | .bcsar              | files   |
	BCSTM  | x | x |   | .bcstm              | WAV     | Yes
	BCWAR  | x |   |   | .bcwav              | files   | 
	BFFNT  | p |   |   | .bffnt              | PNG     | Yes
	BFLAN  | p |   |   | .bflan              | TX      | 
	BFLIM  | x | x |   | .bflim              | PNG     | Yes
	BFLYT  | x |   | x | .bflyt              | TX      |
	CBMD   | e |   |   | .bnr banner[.bin]   | files   |
	DARC   | x | e |   | .arc .bcma (...)    | files   |
	GARC   | x |   | x | .garc (none)        | files   |
	GFA    | x |   |   | .gfa                | files   |
	mini   | x |   | x | .mini .bl .wd (none)| files   |
	MSBT   | x |   |   | .msbt               | TX      |
	NARC   | p |   |   | .narc               | files   |
	NCCH   | x |   |   | .app .cxi .cfa      | sections|
	NDS    | x |   |   | .nds                | files   |
	SARC   | x | x |   | .sarc .arc .szs     | files   |

There is also:

*	unpack.ExtHeader:
	*	Unpacks NCCH extended headers (automatic when unpacking an NCCH partition). Needs a specific file name (exheader.bin, extheader.bin or DecryptedExtHeader.bin) to be recognized. Outputs as a TXTree
*	unpack.ExeFS:
	*	Unpacks ExeFS images (automatic when unpacking an NCCH partition). The file must have the .exefs extension or be "exefs.bin" or "DecryptedExeFS.bin" to be recognized
*	unpack.RomFS:
	*	Unpacks RomFS images (automatic when unpacking an NCCH partition). The file must have the .romfs extension or be "romfs.bin" or "DecryptedRomFS.bin" to be recognized

Compressions:

D: Decompressible

C: Compressible

	Compression | D | C | Extensions                   | Uses libkit
	-----------------------------------------------------------------
	LZ10        | x | x | (none) .cmp .l *_LZ.bin .LZ  |
	LZ11        | x | x | (none) .cmp .l *_LZ.bin .LZ  | Yes
	LZH8        | x |   | .cx                          |
	Yaz0        | x |   | .szs                         |
	ETC1        | x |   | Texture compression in BFLIM |

Dependencies
============
To access all 3DSkit functionnalities, you need:

*	Python 3.5+
*	Numpy
*	Pillow (use sudo apt-get install python3-pil or pip3 install pillow)

Pillow is only needed for textures and images (BFLIM, BFFNT). If you don't have it, other modules will work as well.

Contributing
============

Before submitting an issue, take a look at the program's help (-H option), your problem can be normal, or already explained there.
Don't forget to check if the issue has not been already reported, or fixed (check also in the closed ones).
Check also if you have the last commit
Then, if you really found a new issue, you should precise:

*	Your python version
*	The console output (in verbose mode, -v option)
*	Attach the concerned file and if possible and necessary, a screenshot of the problem in the game.
