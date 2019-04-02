# BFLAN format documentation (Binary caFe Layout ANimation)

The BFLAN format is used to describe user interface animations.
BLAN files are usually bundled in SARC or DARC archives that also contain BFLYT and BFLIM files,
that make a stand-alone UI layout.
It's format is also tree-like, and is quite similiar to the BFLYT format

There are several versions of the BFLAN format, associated with the equivalent BFLYT versions.
There are certainly more, but these have been observed :

- 7.1.0 and 7.2.1 on 3DS
- 8.6.0 on Switch

## FLAN Header

Quite similiar to the BFLYT header

- 4s : Magic number ("FLAN")
- H  : Byte-order mark (0xFFFE for little endian and 0xFEFF for big endian)
- H  : Header length (theorically 0x14)
- I  : Format version in format 0xMMmmbbbb for M.m.b
- I  : Entire file size
- H  : Number of sections (always 2 ?)
- H  : Padding ?

## pat1 section (Part Animation Table ?)

The pat1 section seems to be always the first, and some kind of "header" for pai1.
It contains mostly global informations about animations.

- 4s : Magic number ("pat1")
- I  : Section size
- H  : Animations order
- H  : Number of animation groups
- I  : Offset of the animation name relative to the start of the section, links to an ASCII null terminated string
- I  : Offset of the animation group names
- H  : Start of file
- H  : End of file
- B  : Child binding
- 3x : Padding ?

(Animation group names, offset defined above)
	- [28s]: Names, as ASCII null terminated strings padded to 28 bytes (maybe 36 if >= 8.0.0 ?), number defined above

## pai1 section (Part Animation Information ?)

The pai1 section contains the actual animation data

- 4s : Magic number ("pai1")
- I  : Section size
- H  : Frame size
- B  : Bitflags ? [reference needed]
- x  : Padding
- H  : Number of textures
- H  : Number of entries
- I  : Entry offsets table offset, relative to the start of the section
- [I]: Texture names offsets, relative to the start of this table. Link to ASCII null-terminated names.

- (Entry offsets table, offset defined above)
	- [I]: Entries' offsets, relative to the start of the section, number is number of sections, defined above
- (Entries, at offsets defined by the table)
	- 28s: Entry name, as an ASCII null-terminated string padded to 28 bytes
	- B  : Number of tags
	- B  : Animation target type (0 = pane, 1 = material)
	- H  : Padding
	- [I]: Tags offsets, relative to the entry's start (number defined above)
	- (Tags, at offsets defined above)
		- {If target type = 2}
			- I : <unknown>
		- 4s : Magic (indicates tag's type, see below)
		- I  : Number of entries
		- [I]: Entries' offsets relative to the start of the tag, number defined above
		- (Data entries, at offsets defined above)
			- B : Index ?
			- B : Animation target (see below)
			- H : Data type
			- H : Number of key frames
			- H : Padding ?
			- I : Offset to the first key frame, theorically always 0x0C
			- (Key frames, number defined above)
				- {If data type = 2}
					- f : Frame
					- f : Value
					- f : Blend
				- {If data type = 1}
					- f : Frame
					- H : Value
					- H : Padding ?
			- {Only if tag = FLEU ?}
				- I : Entry name length
				- s : Entry name

## Constants

TAGS:
	- FLPA : FLAN PAne SRT
	- FLVI : FLAN VIsibility
	- FLTP : FLAN Texture Pattern
	- FLVC : FLAN Vertex Colors
	- FLMC : FLAN Material Color
	- FLTS : FLAN Texture SRT
	- FLIM : FLAN Image Material ??
	- FLEU : FLAN ????

FLPA TYPE 2 :

- 0 : X translation
- 1 : Y translation
- 2 : Z translation
- 3 : X rotation
- 4 : Y rotation
- 5 : Z rotation
- 6 : X scale
- 7 : Y scale
- 8 : X size
- 9 : Y size

FLVI TYPE 2 :

- 0 : Visibility

FLTP TYPE 2 :

- 0 : Texture pattern

FLVC TYPE 2 :

- 0 : Top left red
- 1 : Top left green
- 2 : Top left blue
- 3 : Top left alpha
- 4 : Top right red
- 5 : Top right green
- 6 : Top right blue
- 7 : Top right alpha
- 8 : Bottom left red
- 9 : Bottom left green
- 10 : Bottom left blue
- 11 : Bottom left alpha
- 12 : Bottom right red
- 13 : Bottom right green
- 14 : Bottom right blue
- 15 : Bottom right alpha
- 16 : Pane alpha

FLMC TYPE 2 :

- 0 : Black color red
- 1 : Black color green
- 2 : Black color blue
- 3 : Black color alpha
- 4 : White color red
- 5 : White color green
- 6 : White color blue
- 7 : White color alpha
- 8 : Texture color blend ratio
- 9 : Tev color 0 red
- 10 : Tev color 0 green
- 11 : Tev color 0 blue
- 12 : Tev color 0 alpha
- 13 : Tev color 1 red
- 14 : Tev color 1 green
- 15 : Tev color 1 blue
- 16 : Tev color 1 alpha
- 17 : Tev color 2 red
- 18 : Tev color 2 green
- 19 : Tev color 2 blue
- 20 : Tev color 2 alpha
- 21 : Tev konstant color 0 red
- 22 : Tev konstant color 0 green
- 23 : Tev konstant color 0 blue
- 24 : Tev konstant color 0 alpha
- 25 : Tev konstant color 1 red
- 26 : Tev konstant color 1 green
- 27 : Tev konstant color 1 blue
- 28 : Tev konstant color 1 alpha
- 29 : Tev konstant color 2 red
- 30 : Tev konstant color 2 green
- 31 : Tev konstant color 2 blue
- 32 : Tev konstant color 2 alpha

FLTS TYPE 2 :

- 0 : U translation
- 1 : V translation
- 2 : Rotation
- 3 : U scale
- 4 : V scale

FLIM TYPE 2 :

- 0 : Rotation
- 1 : U scale
- 2 : V scale

# About

*This is a format documentation originally made by Tyulis for the 3DSkit project.
It is not an absolute reference, and may contain wrong, outdated or incomplete stuff.
Sources used to make this document and contributors are listed below, the rest has been found by personal investigations.
If you find any error, incomplete or outdated stuff, dont't hesitate to open an issue or a pull request in the [3DSkit GitHub repository](https://github.com/Tyulis/3DSkit).
This document is completely free of charge, you can read it, use it, share it, modify it, sell it if you want without any conditions
(but leaving this paragraph and sharing extensions and corrections of this document on the original repository would just be the most basic of kindnesses)

Documentation about the structure of this document is [here](https://github.com/Tyulis/3DSkit/doc/README.md)*

## Credits and sources
- [BenzinU](https://gbatemp.net/threads/benzinu-release.423171/), by Diddy81
- [http://mk8.tockdom.com/wiki/BFLAN_(File_Format)]
- [https://www.3dbrew.org/wiki/CLAN_format]
- [http://wiki.tockdom.com/wiki/BRLAN]
