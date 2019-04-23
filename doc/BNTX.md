# BNTX format documentation (Binary Nx TeXture)

BNTX is a container format for bitmap textures in Switch games.
It is sometimes embedded into other files like BFFNT fonts.

Two versions are known, on Switch only :

- 0.4.0.0
- 0.4.1.0

The differences are currently unknown

## File structure
### BNTX header

- 8s : Magic number ("BNTX\0\0\0\0")
- I  : File version (as 0xMMmmrrbb for MM.mm.rr.bb)
- H  : Byte order mark (0xFFFE -> little endian, 0xFEFF -> big endian)
- B  : Alignment exponent (the texture data start will be aligned to (1 << exponent) bytes)
- B  : Target adress size ?
- I  : File name offset. Points to the file name string in the \_STR section
- H  : Relocation flag (incremented at each relocation, used only at run time)
- H  : First section offset (usually \_STR)
- I  : Relocation table offset
- I  : Full file size in bytes

### Texture container

- 4s : Target platform ("NX  " (NX and two spaces) for Switch, "GEN " (with one space) for computer)
- I  : Number of textures
- Q  : Texture table offset. Points to a table of 64 bits pointer to the BRTI sections for each texture, length is the number of textures
- Q  : BRTD section offset
- Q  : \_DIC section offset
- Q  : Texture memory pool offset
- Q  : Current memory pool pointer (set only at runtime)
- I  : Base memory pool offset (set only at runtime)
- I  : <unknown>

### \_STR section (STRing table)

A table that contains all strings
The first string is always an empty string and is not counted for the number of strings

- 4s : Magic number ("\_STR")
- I  : Next section offset (relative to the start of the section, does not point on \_DIC)
- I  : Section size (including the following \_DIC)
- I  : <unknown>
- I  : Number of strings in the table
- Then the null string (size 0, empty string, so just 0x00000000)
- (Then strings, number is defined just above)
    - H : String length, does not include the terminating null byte
    - n : The null-terminated string

### \_DIC section (DICtionary)

A dictionary used for fast name lookup

- 4s : Magic number ("\_DIC")
- I  : Number of entries

 Then a root entry, that is **not** included in the number of section.
 After the root entry, the actual entries follow

 - (Entry structure, number defined above + the root entry)
    - I : Reference bit
    - H : Left child node index
    - H : Right child node index
    - Q : Key offset, to the stored key name


### BRTI section (BNTX Texture Info)

This section stores informations about a texture

- 4s : Magic number ("BRTI")
- I  : Next section offset
- I  : Section size
- I  : <unknown>
- H  : <unknown>
- H  : Tile mode (0 = swizzled, 1 = not swizzled)
- H  : Swizzle value
- H  : Number of mipmaps
- H  : Number of multi-samples
- H  : <unknown>
- I  : Image format (see TEXTURE FORMATS below)
- I  : GPU access type
- I  : Texture's width in pixels
- I  : Texture's height in pixels
- I  : Texture's depth in pixels (1 if 2D)
- I  : Textures array length (1 if there's no array)
- I  : Block height exponent, the block height (1 << exponent) is used for swizzling
- I  : <unknown>
- 20s: <unknown>
- I  : Total size of mipmap data
- I  : Texture data alignment (usually 0x200)
- 4B : Respectively, red, green, blue and alpha chennels sources (see CHANNEL SOURCES below)
- B  : Texture dimension (see TEXTURE DIMENSIONS below)
- 3B : <unknown>
- q  : Texture's name offset
- q  : Texture container offset (see above)
- q  : Texture data levels table offset, points on an array of 64-bits integers that point to the different levels of the texture
- q  : User data offset
- q  : Texture pointer, used only at runtime
- q  : Texture view pointer, used only at runtime
- q  : Descriptor slot data offset, used only at runtime
- q  : User data dictionary offset, points to a \_DIC section containing the user data's names

### BRTD section (BNTX Texture Data)

This section holds the actual texture data. Each texture is aligned to the alignment defined in its BRTI and stored contiguously

- 4s : Magic number ("BRTD")
- I  : Unknown
- Q  : Section size (including the header)

### User data

This is a structure used by developers to store data in the file to be used by the program.

- q : Offset of the name of this user data entry
- q : Offset of the data
- I : Number of data entries
- B : Data type : 0x00 -> int32, 0x01 -> float32, 0x02 -> string, 0x03 -> byte

- (Actual data, offset defined above)
    Just a table of values, type and number of elements are defined above

### \_RLT section (ReLocation Table)

- 4s : Magic number ("\_RLT")
- I  : Offset of this table
- I  : Number of sections
- I  : <unknown>
- (Table of sections entry, number of entries defined above)
    - q : Section pointer, set only at runtime
    - I : Section offset
    - I : Section size
    - I : Entry ID
    - I : Number of entries in the section

- (Entry, number and offsets defined above)
    - I : Entry offset
    - H : Array count
    - B : Offset count
    - B : Padding size

## Constants

The texture format is in two bytes, it is actually split in two parts : The first byte is the pixel format, the second is the color space

TEXTURE FORMATS (first byte) :

- 0x02 : L8
- 0x07 : RGB565
- 0x09 : RG8
- 0x0A : L16
- 0x0B : RGBA8
- 0x0F : R11G11B10
- 0x14 : L32
- 0x1A : BC1
- 0x1B : BC2
- 0x1C : BC3
- 0x1D : BC4
- 0x1E : BC5
- 0x1F : BC6H
- 0x20 : BC7
- 0x2D : ASTC4x4
- 0x2E : ASTC5x4
- 0x2F : ASTC5x5
- 0x30 : ASTC6x5
- 0x31 : ASTC6x6
- 0x32 : ASTC8x5
- 0x33 : ASTC8x6
- 0x34 : ASTC8x8
- 0x35 : ASTC10x5
- 0x36 : ASTC10x6
- 0x37 : ASTC10x8
- 0x38 : ASTC10x10
- 0x39 : ASTC12x10
- 0x3A : ASTC12x12

TEXTURE COLOR FORMAT (second byte):

- 0x01 : UNorm
- 0x02 : SNorm
- 0x03 : UInt
- 0x04 : SInt
- 0x05 : Single
- 0x06 : SRGB
- 0x0A : UHalf

TEXTURE DIMENSIONS :

- 0x00 : 1D
- 0x01 : 2D
- 0x02 : 3D
- 0x03 : Cube
- 0x04 : 1D array
- 0x05 : 2D array
- 0x06 : 2D multi-sample
- 0x07 : 2D multi-sample array
- 0x08 : Cube array

CHANNEL SOURCES :

- 0x00 : Zero
- 0x01 : One
- 0x02 : Red
- 0x03 : Green
- 0x04 : Blue
- 0x05 : Alpha

## About

*This is a format documentation originally made by Tyulis for the 3DSkit project, mostly based on the sources below.
It is not an absolute reference, and may contain wrong, outdated or incomplete stuff.
Sources used to make this document and contributors are listed below, the rest has been found by personal investigations.
If you find any error, incomplete or outdated stuff, dont't hesitate to open an issue or a pull request in the [3DSkit GitHub repository](https://github.com/Tyulis/3DSkit).
This document is completely free of charge, you can read it, use it, share it, modify it, sell it if you want without any conditions
(but leaving this paragraph and sharing extensions and corrections of this document on the original repository would just be the most basic of kindnesses)

Documentation about the structure of this document is [here](https://github.com/Tyulis/3DSkit/doc/README.md)*

## Credits and sources
- [https://www.vg-resource.com/thread-31389.html]
- [https://avsys.xyz/wiki/BNTX_(File_Format)]
- [https://github.com/gdkchan/BnTxx]
