**v1.0**

*	v1.0.0:
	*	Initial release (basic pluggable program)

**v1.1**

*	v1.1.0:
	*	Added unpack.SARC
*	v1.1.1:
	*	Added support for SARCs without files names
	*	Fixed some minor console display issues with packed SARCs due to padding issue

**v1.2**

*	v1.2.1:
	*	Added unpack.DARC and introduced rawutil
	*	rawutil is a module based on struct to handle binary data. It can be used as struct with strings as structures (but wit new elements), or as a TypeReader or TypeWriter object
*	v1.2.2:
	*	Many fixes in rawutil (TypeReader didn't work...)
*	v1.2.3:
	*	Fixed some errors in unpack.DARC, like encoding errors, offsets mismatch or issues due to padding
*	v1.2.4:
	*	Updated rawutil

**v1.3**

*	v1.3.4:
	*	Added pack.SARC and pack.DARC
*	v1.3.5:
	*	Fixed some issues in pack.DARC (other encoding issues and padding errors)

**v1.4**

*	v1.4.5:
	*	Added LZ10 and LZ11 decompression
*	v1.4.6:
	*	Fixed some issues in compress.LZ11

**v1.5**

*	v1.5.6:
	*	Imported ALYTtool content:
	*	Added unpack.BFLYT
*	v1.5.7:
	*	Added pack.BFLYT
*	v1.5.8:
	*	Added unpack.ALYT
	*	Removed usage of formats list in unpack module loading
*	v1.5.9:
	*	Fixes in filesystem utilities (util.fileops)
	*	Updated rawutil

**v1.6**

*	v1.6.9:
	*	Added unpack.CBMD
*	v1.6.10:
	*	Added extraction of LTBL section in unpack.ALYT
*	v1.6.11:
	*	Fixed some rawutil issues with int24 and uint24

**v1.7**

*	v1.7.12:
	*	Added unpack.BCSAR (partial)
*	v1.7.13:
	*	Fixed an issue in pack.BFLYT (pane sections order)

**v1.8**

*	v1.8.14:
	*	Added unpack.BCLYT (partial)
*	v1.8.15:
	*	Little extension of the BCLYT support

**v1.9**

*	v1.9.15:
	*	Added unpack.BFLIM (partial)
	v1.9.16:
	*	Fixed some swizzling issues in unpack.BFLIM

**v1.10**

*	v1.10.16:
	*	Added unpack.GARC
	*	Moved format detection in a new module to avoid recursive imports issue in unpack.GARC
*	v1.10.17:
	*	Added support for version 6 GARCs
*	v1.10.18:
	*	Updated rawutil

**v1.11**

*	v1.11.18:
	*	Added LZH8 and Yaz0 decompression
	*	Fixed issues with int24 in rawutil

**v1.12**

*	v1.12.18:
	*	Added NDS ROMs extraction
*	v1.12.19:
	*	Added support for DSi-specific content
*	v1.12.20:
	*	Updated rawutil
*	v1.12.21:
	*	Fixed some filesystem issues in NitroROM FS extraction

**v1.13**

*	v1.13.21:
	*	Added the plugin system (-g options)
	*	Added FULL support for GFA (GFAC) archives extraction
*	v1.13.22:
	*	Fixed errors in pluginCompiler
	*	Fixed submodule loading issue
*	v1.13.23:
	*	Added plugins.readdata
	*	Added plugins.breaddata
*	v1.13.24:
	*	Fixed readme
	*	Fixed options help
	*	Added plugins help

**v1.14**

*	v1.14.24:
	*	Added pack.ALYT
	*	Fixed a minor padding issue in unpack.SARC
	*	Updated rawutil
	*	Fixed plugin loading issue when other options are specified <br>
*	v1.14.25:
	*	Fixed some bugs in color packing by rawutil
	*	Fixed some bugs in pack.BFLYT (mat1 offsets table length calculation issue, thanks to __ericjwg__, and materials offsets mismatch)
*	v1.14.26:
	*	Fixed a huge bug with {} structures packing in rawutil
	*	Fixed a stupid error in unpack.SARC

**v1.15**

*	v1.15.26:
	*	Completely rewrote unpack.BFLIM for uncompressed textures, now works fine
*	v1.15.27:
	*	Fixed ETC1 decompression in unpack.BFLIM, now full support of these files
*	v1.15.28:
	*	Changed format-specific options system, now use -O option, see detailed help
	*	Fixed a little problem in rawutil
*	v1.15.29:
	*	Fixed some bugs in RGBA5551, RGB565 and RGB8 packing and unpacking in BFLIM
*	v1.15.30
	*	Fixed the last bug in BFLIM packing on non-power of two textures

**v1.16**

*	v1.16.30
	*	Added BL archives extraction
	*	Added the newmod plugin to replace and extend pluginCompiler
*	v1.16.31
	*	Added BL archive repacking
	*	Fixed a stupid error in unpack.BL
*	v1.16.32
	*	Added a substructure replacement function in rawutil
	*	Completed the changelog and the readme

**v1.17**

*	v1.17.32
	*	Added unpack.NCCH
	*	Added unpack.ExtHeader
	*	Added an error codes system and added the errors help
*	v1.17.33
	*	Updated the readme
	*	Improved colors precision for RGB565 and RGBA5551 formats of BFLIM files
	*	Updated rawutil
*	v1.17.34
	*	Added unpack.ExeFS
	*	Small orthographic fixes in the help and the changelog
*	v1.17.35
	*	Added the dochecks option for unpack.NCCH and unpack.ExeFS to check files and sections hashes
	*	Fixed a huge bug which prevents use of uint24 and int24 in unpacking functions of rawutil
*	v1.17.36
	*	Merged:
		*	Syntax fix in unpack.GARC (by ObscenityIB)
		*	Fix of \_alyt\_ folder detection in pack.ALYT (by hlixed)
	*	Fixed the error when the swizzle option is not specified to pack.BFLIM (thanks to hlixed)
	*	Improved the readme

**v1.18**

*	v1.18.36
	*	Added LZ10 compression
	*	Huge upgrade to rawutil, completely rewritten in a faster and more readable way
	*	Added the -v option (verbose)
	*	Added the supported formats table in te readme
	*	Added format recognition for unnamed files in SARCs
	*	Fixed a stupid bug in compress.LZ11.decompressLZ11 which created errors in the decompressed data
	*	Fixed a byte order problem in pack.SARC
*	v1.18.37
	*	Updated rawutil to add transparent handling of file objects
	*	Fixed an offset calculation error in unpack.DARC
	*	Added the use of file objects instead of bytes objects in certain modules. This make them MUCH faster, especially on huge files
	*	Converted unpack.GARC to use file objects
	*	Fixed an error which prevents recognition of file names in pack.SARC
*	v1.18.38:
	*	Optimized unpack.BL, unpack.SARC and unpack.ALYT

**v1.19**

*	v1.19.38
	*	Added LZ11 compression
	*	Added pack.GARC
	*	Fixed a stupid bug in rawutil which prevents use of TypeWriter.pack() with file-like objects
*	v1.19.39
	*	Updated help and readme
	*	Several fixes in packed files compression
	*	Prepared use of file-like objects in compressors and decompressors
*	v1.19.40
	*	Rewritten the Yaz0 decompressor for more efficience and less errors (thanks to NWPlayer123)
	*	Now prints the help if no arguments are specified
*	v1.19.41
	*	(Very) slightly enhanced the LZ11 decompressor
	*	Optimized unpack.BFLIM
*	v1.19.42
	*	Cleanup
	*	Rewritten the LZ11 compressor. Now, it takes 0.25s instead of 45s to compress 20KB.
*	v1.19.43
	*	Optimized the LZ11 decompressor
	*	Fixed LZ11 compression
	*	Fixed GARC packing and extraction
	*	Style fixes and cleanup
*	1.19.44
	*	Fixed and updated the readme
	*	Fixed the error which occured when the output directory has the same name as an existing file
	*	Several fixes in compress.Yaz0 and rawutil
	*	Fixed the unsupported operation error in unpack.GARC
	*	Further optimization in the LZ11 decompressor (ca. 1%)
*	v1.19.45
	*	Fixed (avoided...) ANOTHER bug in BFLIM extraction
	*	Fixed a little bug in output directory creation
*	v1.19.46
	*	Small fix in txtree
	*	Fixed extraction of some Pokémon USUM GARCs

**v1.20**

*	v1.20.46
	*	Added unpack.BFLAN
	*	Minor fixes
*	v1.20.47
	*	Minor fixes
	*	Little verbosity increase (now says the file name)
	*	Fixed OSError on too short files
	*	Fixed BFLIM transpose (8) swizzling orientation
	*	Little fix in ETC1A4 decompression
	*	Fixed a problematic bug in LZ11 decompression (unexpected zeroes)
	*	Added error handling for empty files
*	v1.20.48
	*	Fixed the bug with the rotation on deswizzling of BFLIMs
	*	Fixed a bug with absolute path in pack.GARC

**v1.21**

*	v1.21.48
	*	Now, when packing without the -o option, 3DSkit will determine an output name from the input name and the format
	*	Fixed multiple bugs in pack.BFLIM
	*	Fixed pack.ALYT and pack.SARC padding issues
*	v1.21.49
	*	Fixed unpack.BFLYT
	*	Enhanced txtree by removing all single-quotes and 'list' or 'tuple' mentions in output files
*	v1.21.50
	*	Fixed pack.BFLYT
	*	Using -x on a directory now unpacks all contained files recursively
	*	Cleanup
	*	Fixes in txtree
	*	Updated rawutil
*	v1.21.51
	*	Added a warning system (error code 9XX)
	*	Fixed a bug when recursively extract a folder containing different file formats
	*	Improved readability of console output when recursively extracting a directory
*	v1.21.52
	*	"Fixed" the compression recognition bug in unpack.GARC
	*	Added support for all mini packed files, not only BL

**v1.22**

*	v1.22.52
	*	Added unpack.BCSTM
	*	Minor fixes
*	v1.22.53
	*	Fixed the frame rate issue in unpack.BCSTM
	*	Updated the help and the README

**v1.23**

*	v1.23.53
	*	Cleanup
	*	Started unpack.NARC
	*	Fixed some bad compression recognition issues
	*	Huge refactoring of the error/warning function, now much easier to use (internal)
	*	Added support for custom magics at mini files repacking
	*	Updated the README
*	v1.23.54
	*	Fixed some stupid errors in pack.mini
	*	Added unpack.NARC (partial)
	*	Now empty file don't cause any more errors in recursive extraction
	*	Added EM in recognized mini magics
*	v1.23.55
	*	Fixed the non-extraction of the last file in unpack.mini
*	v1.23.56
	*	A little internal refactoring
	*	Fixed a byte order issue in rawutil
	*	Added a quiet (-q) option (no console output)
	*	Added a debug (-V) option
*	v1.23.57
	*	You can now give the output file name to the -C and -D (compression/decompression) options
	*	Rewritten the LZ10 compression algorithm like the LZ11 one (~15% less compression, ~20x performance)
	*	Updated the readme and the help, added some examples in the README

**v1.24**

*	v1.24.57
	*	Finally added the RomFS support, in a quite well optimized way (no annoying temporary 3GB file...)
*	v1.24.58
	*	Added handling of KeyboardInterrupts (error 204)
	*	Fixed the NCCH and ExeFS hash check
	*	NCCH and ExeFS hash checks are now done by default (use -O{dochecks=false} if you don't want to do them)
	*	Rewritten the format recognition function in a less hacky way
	*	Fixed the non-recognition of ExeFS and RomFS files
	*	Fixed compression recognition
	*	Added some little scripts I use for the dev, if it can interest someone (util/dev_scripts)
*	v1.24.59
	*	Optimized unpack.GFA and unpack.NDS
	*	Fixed a byte order error in unpack.NARC
*	v1.24.60
	*	Fixed NCCH's ExeFS extraction
	*	Updated unpack.ExtHeader
