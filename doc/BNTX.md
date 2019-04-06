# BNTX format documentation (Binary Nx TeXture)

BNTX is a container format for bitmap textures in Switch games.
It is sometimes embedded into other files like BFFNT fonts.

The only known version is 0.4.0.0

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
- Q  : Textures' data offset
- Q  : \_DIC section offset
- Q  : Texture memory pool offset
- Q  : Current memory pool pointer (set only at runtime)
- I  : Base memory pool offset (set only at runtime)
- I  : <unknown>

### BRTI section (Texture Info)

This section stores informations about a texture

- 4s : Magic number ("BRTI")
- I  : Next section offset
- I  : Section size
- I  : <unknown>
- H  : <unknown>
- H  : Tile mode (0 = swizzled, 1 = not swizzled)
- H  : Swizzle value
- H  : Number of images
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
- q  : Texture table offset (same use as the Texture Container's field)
- q  : User data offset
- q  : Texture pointer, used only at runtime
- q  : Texture view pointer, used only at runtime
- q  : Descriptor slot data offset, used only at runtime
- q  : User data dictionary offset, points to a \_DIC section containing the user data's names

### \_STR section (STRing table)

A table that contains all strings
The first string is always an empty string and is not counted for the number of strings

- 4s : Magic number ("\_STR")
- I  : Next section offset
- I  : Section size
- I  : <unknown>
- I  : Number of strings in the table
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

### User data

This is a structure used by developers to store data in the file to be used by the program.

- q : Offset of the name of this user data entry
- q : Offset of the data
- I : Number of data entries
- B : Data type : 0x00 -> int32, 0x01 -> float32, 0x02 -> string, 0x03 -> byte

- (Actual data, offset defined above)
    Just a table of values, type and number of elements are defined above

### \-RLT section (ReLocation Table)

- 4s : Magic number ("\_RLT")
- I  : Offset of the table start
- I  : Number of sections
- I  : <unknown>
- (Table of sections entry, offset and number of entries defined above)
    - q : Section pointer, set only at runtime
    - I : Section offset
    - I : Section size
    - I : Entry ID
    - I : Number of entries in the section

- (Section)
    - I : Entry offset
    - H : Array count
    - B : Offset count
    - B : Padding size

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
