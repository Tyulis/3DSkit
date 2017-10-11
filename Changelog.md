v1.0.0:
	Initial release (basic pluggable program)

v1.1.0:
	Added unpack.SARC
v1.1.1:
	Added support for SARCs without files names
	Fixed some minor console display issues
	
v1.2.1:
	Added unpack.DARC and introduced rawutil
	rawutil is a module based on struct to handle binary data. It can be used as struct with strings as structures (but wit new elements), or as a TypeReader or TypeWriter object
v1.2.2:
	Many fixes in rawutil (TypeReader didn't worked)
v1.2.3:
	Fixed some errors in unpack.DARC, like encoding errors, offsets mismatch or issues dued to padding
v1.2.4:
	Updated rawutil

v1.3.4:
	Added pack.SARC and pack.DARC
v1.3.5:
	Fixed some issues in pack.DARC (other encoding issues and padding errors)

v1.4.5:
	Added LZ10 and LZ11 decompression
v1.4.6:
	Fixed some issues in compress.LZ11

v1.5.6:
	Imported ALYTtool content:
	Added unpack.BFLYT
v1.5.7:
	Added pack.BFLYT
v1.5.8:
	Added unpack.ALYT
	Removed use of formats list in unpack module loading
v1.5.9:
	Fixes in filesystem utilities
	Updated rawutil

v1.6.9:
	Added unpack.CBMD
v1.6.10:
	Added extraction of LTBL section in unpack.ALYT
v1.6.11:
	Fixed some rawutil issues with uint24

v1.7.12:
	Added unpack.BCSAR (partial)
v1.7.13:
	Fixed an issue in pack.BFLYT (pane sections order)

v1.8.14:
	Added unpack.BCLYT (partial)
v1.8.15:
	Little extension of the BCLYT support

v1.9.15:
	Added unpack.BFLIM (partial)
v1.9.16:
	Fixed some swizzling issues in unpack.BFLIM

v1.10.16:
	Added unpack.GARC
	Moved format detection in a new module to avoid recursive imports issue in unpack.GARC
v1.10.17:
	Added support for version 6 GARCs
v1.10.18:
	Updated rawutil

v1.11.18:
	Added LZH8 and Yaz0 decompression
	Fixed int24 issues in rawutil

v1.12.18:
	Added NDS ROMs extraction
v1.12.19:
	Added support for DSi-specific content
v1.12.20:
	Updated rawutil
v1.12.21:
	Fixed some filesystem issues in NitroROM FS extraction
	
v1.13.21:
	Added the plugin system (-g options)
	Added FULL support for unpacking GFA (GFAC) archives
v1.13.22:
	Fixed errors in pluginCompiler
	Fixed submodule loading issue
v1.13.23:
	Added plugins.readdata
	Added plugins.breaddata
v1.13.24:
	Fixed readme
	Fixed options help
	Added plugins help

v1.14.24:
	Added pack.ALYT
	Fixed a minor padding issue in unpack.SARC
	Updated rawutil
	Fixed plugin loading issue when other options are specified
v1.14.25:
	Fixed some bugs in color packing by rawutil
	Fixed some bugs in pack.BFLYT (mat1 offsets table length calculation issue, thanks to ericjwg, and materials offsets mismatch)
