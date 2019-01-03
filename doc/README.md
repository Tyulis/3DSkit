# Documentation

In this folder will be placed documentation about file formats handled by 3DSkit

Note that this is probably not complete nor perfect, and some things have been found by home made retro-engineering.
There are also sometimes several versions of the same format, so be careful.

If you find anything that is not there, an error, or something outdated, don't hesitate to open an issue or a pull request, any help is welcome :)

## Structures

It those documents, the data structures are explained in a specific way, to help comprehension and writing.
This is the doc about the documents' scheme. If you know the structure format for the python module `struct`, or better `rawutil`,
you mostly know how to read this.

Sections and chunks of the file will be denoted by Markdown titles. If there is an order in the file, it is theorically respected in the doc.
Here is an example section doc with its explanation :

### Section name (usually with its meaning if known)

Description of the section

- 4s : String, 4 bytes long (string, bytes, char[], 1o per character)
- B  : Unsigned 8-bits integer (uint8 / ubyte, 1o)
- b  : Signed 8-bits integer (int8 / byte / char, 1o)
- H  : Unsigned 16-bits integer (uint16 / ushort, 2o)
- h  : Signed 16-bits integer (int16 / short, 2o)
- I  : Unsigned 32-bits integer (uint32 / uint, 4o)
- i  : Signed 32-bits integer (int32 / int, 4o)
- Q  : Unsigned 64-bits integer (uint64, 8o)
- q  : Signed 64-bits integer (int64, 8o)
(Group of data, may be conditional or in several copies, or at an other place in the file)
	- f : 32-bits single-precision float (float, 4o)
	- d : 64-bits double-precision float (double, 8o)
{Differences between several possibilities, usually related to file format versions}
	- ? : 8-bits boolean, typically 0 or 1 (bool / uint8_t, 1o)
{Other possibility}
	- [I]: List, number of elements is usually precised.
- 4B : Group of 4 related uint8 (a color, for example)
- n  : String of arbitrary size, terminated by a null byte
- B  : Example of bitflags
	- Bit 0-2 (0b00000111) : Bytes are counted from least to most significant bit, with the associated bit mask
	- Bit 3   (0b00001000) : <unknown> (indicates something unknown, to be completed)
*<Some other information, for example a reference to another part or an unknown structure>*
