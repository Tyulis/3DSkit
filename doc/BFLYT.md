# BFLYT format documentation (Binary caFe LaYouT)

The BFLYT format is a file format used to describe user interface layouts.
As its name indicate it, it was originally a format designed for the WiiU,
but it is also used in several 3DS and Switch games.

BFLYT files are usually bundled with textures (in BFLIM format) and animations (in BFLAN format),
in a standalone archive (usually an SARC or DARC archive)

This file format has a wide number of versions, this ones have been observed :

- On WiiU, version 5.2.0
- On 3DS, versions 7.1.0, 7.2.0 and 7.2.1
- On Switch, version 8.6.0

There are certainly more versions of the format

## FLYT Header

- 4s : Magic number ("FLYT")
- H  : Byte-order mark (0xFFFE -> little endian, 0xFEFF -> big endian)
- H  : FLYT header length (theorically 0x14)
- I  : Format version, in format 0xMMmmrrbb for MM.mm.rr.bb
- I  : Complete file size
- H  : Number of sections
- H  : Padding

## Data sections

Data sections follow just after the FLYT header, and are contiguous.
Every section is aligned to have a size multiple of 4 bytes.

### lyt1 section (LaYouT)

This sections contains several meta data about the layout, and usually follow the header

- 4s : Magic number ("lyt1")
- I  : Section size
- B  : Drawn from middle (boolean)
- 3B : Padding
- f  : Screen width
- f  : Screen height
- f  : Max parts width
- f  : Max parts height
- n  : Layout name

### txl1 section (TeXture List)

Contains links to the related textures (usually bundled with layouts in an archive, in BFLIM format)

- 4s : Magic number ("txl1")
- I  : Section size
- I  : Number of textures
- [I]: Offset table, contains offsets of null-terminated strings relative to the beginning of the table
- [n]: Texture file names

### fnl1 section (FoNt List)

Contains links to the related fonts (usually somewhere else, in BFFNT format). The structure is similiar with txl1

- 4s : Magic number ("fnl1")
- I  : Section size
- I  : Number of textures
- [I]: Offset table, contains offsets of null-terminated strings relative to the beginning of the table
- [n]: Font file names

### mat1 section (MATerials)

Defines the materials used in the layout.

- 4s : Magic number ("mat1")
- I  : Section size
- I  : Number of materials
- [I]: Offset table, contains offsets to meterial descriptions, relative to the beginning of the section
- [Material descriptions]:
	- 28s : Material name, as a null-terminated string padded with zeros to 20 bytes
	- {For versions < 8.0.0}
		- 4B  : Foreground color, as an RGBA quadruplet
		- 4B  : Background color, as an RGBA quadruplet
		- I   : Bitflags (when there is several textures, each information of the same type is probably for its respective texture)
			- Bit 0-1   (0x00000003): Number of texture references
			- Bit 2-3   (0x0000000c): Number of texture transformations
			- Bit 4-5   (0x00000030): Number of mapping settings
			- Bit 6-7   (0x000000C0): Number of texture combiners
			- Bit 9     (0x00000200): Has alpha comparison conditions
			- Bit 10-11 (0x00000C00): Number of blending modes
			- Bit 12-13 (0x00003000): Number of alpha blending modes
			- Bit 14    (0x00004000): Has indirect adjustment
			- Bit 15-16 (0x00018000): Number of projection mappings
			- Bit 17    (0x00020000): Has shadow blending
	- {For versions >= 8.0.0 **/!\\ Warning : this is a supposition /!\\**}
		- I  : Bitflags (same as above)
		- 4B : Unknown
		- 4B : Foreground color, as an RGBA quadruplet
		- 4B : Background color, as an RGBA quadruplet
	- (Texture references, number defined above)
		- H : File name index in txl1
		- B : Wrap S (index in WRAPS, see below for constants)
		- B : Wrap T (index in WRAPS, see below for constants)
	- (Texture transformations, number defined above)
		- f : X translation
		- f : Y translation
		- f : Rotation
		- f : X scale
		- f : Y scale
	- (Mapping settings, number defined above)
		- B  : <unknown>
		- B  : Mapping method (index in MAPPING METHODS, see below for constants)
		- {For versions < 8.0.0}
			- 6B : <unknown>
		- {For versions >= 8.0.0}
			- 14B : <unknown>
	- (Texture combiners, number defined above)
		- B  : Color blending (index in COLOR BLENDING, see below for constants)
		- B  : Alpha blending (index in ALPHA BLENDING, see below for constants)
		- 2B : <unknown>
	- (Alpha comparison, as defined above)
		- B  : Condition (index in ALPHA COMPARISONS, see below for constants)
		- 3B : <unknown>
		- f  : Value
	- (Blending mode, number defined above)
		- B : Operation (index in BLENDING OPERATIONS, see below for constants)
		- B : Source blending (index in BLENDING CALC, see below for constants)
		- B : Destination blending (index in BLENDING CALC, see below for constants)
		- B : Logical operation (index in BLENDING LOGICAL, see below for constants)
	- (Alpha blending mode, number defined above)
		- B : Operation (index in BLENDING OPERATIONS, see below for constants)
		- B : Source blending (index in BLENDING CALC, see below for constants)
		- B : Destination blending (index in BLENDING CALC, see below for constants)
		- B : Logical operation ? (index in BLENDING LOGICAL, see below for constants) [reference needed]
	- (Indirect adjustment, as defined above)
		- f : Rotation
		- f : X warp
		- f : Y warp
	- (Projection mappings, number defined above)
		- f  : X translation
		- f  : Y translation
		- f  : X scale
		- f  : Y scale
		- B  : Option (index in MAPPING OPTIONS, see below for constants)
		- 3B : <unknown>
	- (Shadow blending, as defined above)
		- 3B : Black blending, as an RGB triplet
		- 4B : White blending, as an RGBA quadruplet
		- B  : Probably padding

## Pane tree

After these sections, there is the *pane tree*, so each sections defines a *pane*, an interface object.
There is always a root pane called "RootPane". Each node is defined like this :

- Any pane section with the node's information (typically a pan1)
- A pas1 section to begin the node
- *Children panes*
- A pae1 section to terminate the pane

So each pane (or node) between the pas1 and the related pae1 section is a child of the base pane, it's basically a tree.

Every pane section (so everything in this part of the doc except pas1 and pae1) contains a *"pane data section"*,
which has always the same structure for all pane types :

- B  : Bitflags
	- Bit 0 (0b00000001) : Visible
	- Bit 1 (0b00000010) : Transmit alpha to children
	- Bit 2 (0b00000100) : Has position adjustment
	- *<Other bits : probably unused [reference needed]>*
- B  : Origin flags
	- Bit 0-1 (0b00000011) : X Origin (index in X ORIGIN, see below for constants)
	- Bit 2-3 (0b00001100) : Y Origin (index in Y ORIGIN, see below for constants)
	- Bit 4-5 (0b00110000) : Parent X Origin (index in X ORIGIN, see below for constants)
	- Bit 6-7 (0b11000000) : Parent Y Origin (index in Y ORIGIN, see below for constants)
- B  : Pane's alpha value
- B  : Pane's scale
- 32s: Pane's name, as a null terminated string, padded with zeros to 32 bytes
- f  : X translation
- f  : Y translation
- f  : Z translation
- f  : X rotation
- f  : Y rotation
- f  : Z rotation
- f  : X scale
- f  : Y scale
- f  : Pane width
- f  : Pane height

### pan1 section (PANe)

This defines a "null" pane, an invisible abstract pane with only an pane meta-data section. It is usually used for tree nodes, while "concrete" panes are usually leaves of the tree

- 4s : Magic number ("pan1")
- I  : Section size
- *\<Pane data section\>*

### pas1 section (PAne Start)

A little section that indicates the start of the children list

- 4s : Magic number ("pas1")
- I  : Section size (theorically 8)

### pae1 section (PAne End)

A little section that indicates the end of a tree node

- 4s : Magic number ("pae1")
- I  : Section size (theorically 8)

### bnd1 section (BouNDary)

This defines a "boundary", and is basically a pane without any additional information. Purpose unknown, maybe something to hide children outside of it.
This is always a node pane

- 4s : Magic number ("pan1")
- I  : Section size
- *\<Pane data section\>*

### wnd1 section (WiNDow)

This defines a "window", where other panes are shown. This is usually a node.

- 4s : Magic number ("wnd1")
- I  : Section size
- *\<Pane data section\>*
- H  : Left stretch
- H  : Right stretch
- H  : Top stretch
- H  : Bottom stretch
- H  : Custom left
- H  : Custom right
- H  : Custom top
- H  : Custom bottom
- B  : Number of frames
- B  : Bitflags ?
- H  : Padding
- I  : Content offset, relative to the start of the section
- I  : Frame offset table offset, relative to the start of the section

(Window content, offset defined above)
	- 4B : Top left vertex color (RGBA)
	- 4B : Top right vertex color (RGBA)
	- 4B : Bottom left vertex color (RGBA)
	- 4B : Bottom right vertex color (RGBA)
	- H  : Material index
	- B  : Number of UV coordinates
	- B  : Padding ?
	(UV coordinates structure, number defined above)
		- 2f : Top-left UV coordinates
		- 2f : Top-right UV coordinates
		- 2f : Bottom-left UV coordinates
		- 2f : Bottom-right UV coordinates

(Window frame table, offset defined above)
	- [I] : Offset of window frame structures relative to the start of the section, number defined above

(Window frame structure, offsets defined above)
	- H : Material index
	- B : Texture flip
	- B : Padding

### txt1 section (TeXT)

Defines a pane displaying text

- 4s : Magic number ("txt1")
- I  : Section size
- *\<Pane data section\>*
- H  : Text length
- H  : Restricted text length
- H  : Material index
- H  : Font index
- B  : Text alignment flags
	- Bit 0-1 (0b00000011) : Horizontal alignment (index in X ORIGIN, see below for constants)
	- Bit 2-3 (0b00001100) : Vertical alignment (index in Y ORIGIN, see below for constants)
- B  : Line alignment (index in TEXT ALIGNMENT, see below for constants)
- B  : Bitflags
	- Bit 0 (0b00000001) : Shadow enabled
	- Bit 1 (0b00000010) : Restricted text length enabled
	- Bit 2 (0b00000100) : Invisible border
	- Bit 3 (0b00001000) : Two-cycles border rendering
	- Bit 4 (0b00010000) : Per character transform enabled
- B  : Padding ?
- f  : Italic tilt
- I  : Text offset, relative to the start of the section. The text is null terminated (theorically of the defined length), and encoded in UTF-16
- 4B : Font top color (RGBA)
- 4B : Font bottom color (RGBA)
- f  : X font size
- f  : Y font size
- f  : Character spacing
- f  : Line spacing
- I  : Textbox name offset (*"call name"*), relative to the start of the section, if non zero. The call name is null terminated and encoded in ASCII
- 2f : Shadow position (X, Y)
- 2f : Shadow size (X, Y)
- 4B : Shadow top color (RGBA)
- 4B : Shadow bottom color (RGBA)
- f  : Shadow italic tilt ? [reference needed]
- I  : Per character transform structure offset, relative to the start of the section, if non zero

(Per character transform structure, offset defined above)
<unknown>

### pic1 section (PICture)

Defines a pane that displays an image

- 4s : Magic number ("pic1")
- I  : Section size
- *\<Pane data section\>*
- 4B : Top-left vertex color (RGBA)
- 4B : Top-right vertex color (RGBA)
- 4B : Bottom-left vertex color (RGBA)
- 4B : Bottom-right vertex color (RGBA)
- H  : Material index
- B  : Number of UV coordinates
- B  : Padding
- (UV coordinates, number defined above)
	- 2f : Top-left UV coordinates
	- 2f : Top-right UV coordinates
	- 2f : Bottom-left UV coordinates
	- 2f : Bottom-right UV coordinates

### prt1 section (PaRT ?)

**/!\ Some things here are not sure nor complete /!\\**
This is a generic pane that contains other panes. Purpose unknown, maybe used to group panes for batch operations ?
Note that these sub-sections are not counted by the number of sections in the main header

- 4s : Magic number ("prt1")
- I  : Section size (including contained panes)
- *\<Pane data section\>*
- I  : Number of entries
- 2f : Sections scale (X, Y)

(Sub-panes table entries, number defined above)
	- 24s: Entry name, null terminated and padded with zeros to 24 bytes
	- B  : <unknown>
	- B  : Bitflags ?
	- H  : Padding ?
	- I  : Sub-pane offset, relative to the start of the section. If non-zero, links to the pane data, that is just an usual section, may be pic1, wnd1, bnd1, txt1 or prt1
	- I  : Complementary data offset, relative to the start of the section. If non-zero, links to an usd1 sub-section
	- I  : Extra data offset, relative to the start of the section. If non-zero, links to an "extra" data structure 48 bytes long, with an unknown structure.

{Observed only with version >= 8.0.0, conditions not sure}
	- 24s : Part name ?

## Group tree

After the pane tree, there is another tree on the same principle, that indicates pane groups, probably for batch operations.
It works exactly same as the pane tree, with :

- A grp1 section that contains informations about the group
- A grs1 section to start the children list
- Children groups
- And a gre1 section to end the list.

There is always a root group called "RootGroup".

### grp1 section (GRouP)

Contains informations about a group

- 4s : Magic number ("grp1")
- I  : Section size
- {For version <= 0x05020000 (5.2.0)}
	- 24s: Group name, null-terminated and padded with zeros to 24 bytes (0x18)
	- H  : Number of children
	- H  : Padding
- {For version > 0x05020000 (5.2.0)}
	- 34s: Group name, null-terminated and padded with zeros to 34 bytes (0x22)
	- H  : Number of children
- [24s] : Children panes names, number defined above, each name is null terminated and padded with zeros to 24 bytes (0x18)

### grs1 section (GRoup Start)

Starts the list of children groups of a group

- 4s : Magic number ("grs1")
- I  : Section size (theorically 8)

### gre1 section (GRoup End)

Ends the list of children groups of a group

- 4s : Magic number ("gre1")
- I  : Section size (theorically 8)

## cnt1 section

**/!\ Some things here are not sure nor complete /!\\**

This section is usually the last in the file, and contains information related to animations

- 4s : Magic number ("cnt1")
- I  : Section size
- I  : Section name offset, relative to the start of the section. Point on a null terminated ASCII name aligned to have a length multiple of 4
- I  : Main table offset, relative to the start of the section.
- H  : Number of parts
- H  : Number of animations
- I  : Parts table offset, relative to the start of the section.
- I  : Animations table offset, relative to the start of the section ? [supposition, maybe one table by part]
- n : Duplicate of the section name ?

(Main table, offset defined above)
	- [24s]: Parts names, as ASCII null terminated strings padded by zeros to 24 bytes
	- [I]  : Animation name offsets relative to the start of this table, number is number of animations defined above
	- [n]  : Animation names, as null-terminated ASCII strings at defined offsets

(Parts table, offset defined above)
	- [I]  : Name offsets, relative to the start of this table. Number is the number of parts
	- [n]  : Parts names, as ASCII null terminated strings

(Animations table, offset defined above)
	- [I]  : Name offsets, relative to the start of this table. Number is the number of animations
	- [n]  : Animation names, as ASCII null terminated strings

## usd1 sections (USer Defined ?)

usd1 sections give additional data about another section. Content is defined by the creator, and usually just follows the section it is related.
Completed sections are usually panes, but it may also be anywhere in the file, for example just after the lyt1 section.

- 4s : Magic number ("usd1")
- I  : Section size
- H  : Number of entries
- H  : <unknown>
- (Entries, number defined above)
	- I : Entry's name offset, relative to the start of the entry. Links to the name encoded in ASCII, null terminated.
	- I : Data offset, relative to the start of the entry
	- H : Number of values or length of the value, depending of the data type
	- B : Data type, see below.
	- B : <unknown>
	- {For data type 0, offset defined above}
		- s : Raw data, usually a string, length defined above
	- {For data type 1, offset defined above}
		- [i]: Signed 32-bits integers, number defined above
	- {For data type 2, offset defined above}
		- [f]: 32-bits single-precision floats, number defined above
	- {For data type 3, offset defined above. Only encountered in Switch files}
		- 2H : <unknown>
		- I  : <unknown>
		- I  : <unknown>. The offset reference is at this position
		- I  : Number of strings
		- [I]: Strings offsets, number defined above, relative to 4 bytes before the number of strings (see above)
		- Then null terminated strings at defined offsets

## Constants

These are the lists of constants used by some fields

WRAPS:

- 0 : Near Clamp
- 1 : Near Repeat
- 2 : Near Mirror
- 3 : GX2 Mirror Once
- 4 : Clamp
- 5 : Repeat
- 6 : Mirror
- 7 : GX2 Mirror Once Border

MAPPING METHODS :

- 0 : UV Mapping
- 3 : Orthogonal Projection
- 4 : Pane Based Projection

ALPHA BLENDING :

- 0 : Max
- 1 : Min

COLOR BLENDING :

- 0 : Overwrite
- 1 : Multiply
- 2 : Add
- 3 : Exclude
- 5 : Subtract
- 6 : Dodge
- 7 : Burn
- 8 : Overlay
- 9 : Indirect
- 10 : Blend Indirect
- 11 : Each Indirect

ALPHA COMPARISONS :

- 0 : Never
- 1 : Less
- 2 : Less-or-Equal
- 3 : Equal
- 4 : Not-Equal
- 5 : Greater-or-Equal
- 6 : Greater
- 7 : Always

BLENDING CALC :

- 2 : FB Color
- 3 : FB Color (1)
- 4 : Pixel Alpha
- 5 : Pixel Alpha (1)
- 6 : FBAlpha
- 7 : FBAlpha (1)
- 8 : PixelColor
- 9 : PixelColor (1)

BLENDING OPERATIONS :

- 1 : Add
- 2 : Subtract
- 3 : Reverse Subtract
- 4 : Min
- 5 : Max

BLENDING LOGICAL :

- 0 : None
- 1 : No operation
- 2 : Clear
- 3 : Set
- 4 : Copy
- 5 : Invert Copy
- 6 : Invert
- 7 : And
- 8 : Nand
- 9 : Or
- 10 : Nor
- 11 : Xor
- 12 : Equivalent
- 13 : Reverse And
- 14 : Invert And
- 15 : Reverse Or
- 16 : Invert Or

MAPPING OPTIONS :

- 0 : Standard
- 1 : Entire Layout
- 4 : Pane R and S Projection

TEXT ALIGNMENT :

- 0 : Undefined
- 1 : Left
- 2 : Center
- 3 : Right

X ORIGIN :

- 0 : Center
- 1 : Left
- 2 : Right

Y ORIGIN :

- 0 : Center
- 1 : Up
- 2 : Down

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
- [http://mk8.tockdom.com/wiki/BFLYT_(File_Format)]
- [bflyttool](https://github.com/dnasdw/bflyttool) by dnasdw
