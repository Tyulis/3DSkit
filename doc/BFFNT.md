# BFFNT format documentation (Binary caFe FoNT)

The BFFNT is a text font format, originally used on WiiU but also in several Switch and 3DS games

Several versions have been found :

- 3DS : 3.0.0, 4.0.0
- WiiU : 4.0.0
- Switch : 4.1.0

## File structure
### FFNT Header

- 4s : Magic number ("FFNT")
- H  : Byte-order mark (0xFFFE -> Little endian, 0xFEFF -> Big endian)
- H  : FFNT header length (shoud be 0x14)
- I  : File version in format 0xMMmmrrbb for MM.mm.rr.bb
- I  : Full file size
- I  : Number of sections


### FINF section (Font INFo)

This section stores general data about the font. It immediately follows the FFNT header

- 4s : Magic number ("FINF")
- I  : Section size
- B  : Font type
- B  : Height
- B  : Width
- B  : Ascent
- H  : Line feed
- H  : Alter index
- B  : Default left width
- B  : Default glyph width
- B  : Default character width
- B  : Encoding (the only known value is 1 for UTF-16, UTF-32 in Switch files)
- I  : TGLP section offset (Warning : points to the data's start, substract 8 to also get the magic and section size)
- I  : CWDH section offset (Warning : points to the data's start, substract 8 to also get the magic and section size)
- I  : CMAP section offset (Warning : points to the data's start, substract 8 to also get the magic and section size)


### CMAP (Character MAP)

Maps characters to the glyphs on the sheet.
It is sometimes in several sections. If so, the `next offset` field below will point on the start of the next CMAP section's data (-8 to get the header too)
This may be used to build the character map with elements like (character : glyph index on the sheets)

- 4s : Magic number ("CMAP")
- I  : Section size
- {For versions >= 4.1.0}
    - I : Start code (first character mapped in the section)
    - I : End code (last character mapped in the section)
- {For versions < 4.1.0}
    - H : Start code (first character mapped in the section)
    - H : End code (last character mapped in the section)
- H : Mapping method, see below. Can be 0 for Direct, 1 for Table or 2 for Scan
- H : <unknown>
- I : Next CMAP offset. If non-null, points to the next CMAP section (see above). If zero, there's no more CMAPs.

- {If mapping method is Direct (0x0000)}
    - H : Index offset
    Then for each `code` from the `start code` to the `end code`, the glyph index is `code - start code + index offset`
- {If mapping method is Table (0x0001)}
    - [H] : Glyph indices
    There's the glyph index in the table above for each character from the start code to the end code
- {If mapping method is Scan (0x0002)}
    - {For versions >= 4.1.0}
        - I : Number of items in the table
        - (Items, number defined just above)
            - I : Character code
            - H : Corresponding glyph index
            - H : <unknown>, probably padding
    - {For versions < 4.1.0}
        - H : Number of items in the table
        - (Items, number defined just above)
            - H : Character code
            - H : Corresponding glyph index

### CWDH (Character WiDtHs)

Gives the characters' left, glyph and char widths. Has the same multiple sections system as CMAP.

- 4s : Magic number ("CWDH")
- I  : Section size
- H  : Start code, first character in the section
- H  : End code, last character in the section
- I  : Next CWDH section offset. If non-null, points to the start of the next CWDH section's data (-8 to get the header too)
- (Number of occurences is end code - start code)
    *<The corresponding character is index + start code>*
    - b : Left width
    - b : Glyph width
    - b : Character width

### TGLP (Texture GLyPh ?)

Contains the actual glyphs sheets, as regular bitmap textures.

- 4s : Magic number ("TGLP")
- I  : Section size
- B  : Cell width
- B  : Cell height
- B  : Number of sheets
- B  : Max width
- I  : Number of bytes per sheet
- H  : Base line position
- H  : Texture format (see TEXTURE FORMATS below)
- H  : Number of columns
- H  : Number of lines
- H  : Sheet width (in pixels)
- H  : Sheets height (in pixels)
- I  : Texture data offset

Then, in Switch files, the sheets are contained in a classic BNTX section that starts at the texture data offset
In other versions, the sheets' texture data are just contiguous at the data offset


## Constants

**For 3DS files :**

TEXTURE FORMATS :

- 0x00 : RGBA8
- 0x01 : RGB8
- 0x02 : RGBA5551
- 0x03 : RGB565
- 0x04 : RGBA4
- 0x05 : LA8
- 0x06 : RG8
- 0x07 : L8
- 0x08 : A8
- 0x09 : LA4
- 0x0A : L4
- 0x0B : A4
- 0x0C : ETC1
- 0x0D : ETC1A4

**For WiiU files**

TEXTURE FORMATS :

- 0x00: RGBA8
- 0x01: RGB8
- 0x02: RGBA5551
- 0x03: RGB565
- 0x04: RGBA4
- 0x05: LA8
- 0x06: LA4
- 0x07: A4
- 0x08: A8
- 0x09: BC1
- 0x0A: BC2
- 0x0B: BC3
- 0x0C: BC4
- 0x0D: BC5
- 0x0E: RGBA8_SRGB
- 0x0F: BC1_SRGB
- 0x10: BC2_SRGB
- 0x11: BC3_SRGB

**For Switch files**

TEXTURE FORMATS :

- 0x00: RGBA8
- 0x01: RGB8
- 0x02: RGBA5551
- 0x03: RGB565
- 0x04: RGBA4
- 0x05: LA8
- 0x06: LA4
- 0x07: A4
- 0x08: A8
- 0x09: BC1
- 0x0A: BC2
- 0x0B: BC3
- 0x0C: BC4
- 0x0D: BC5
- 0x0E: RGBA8_SRGB
- 0x0F: BC1_SRGB
- 0x10: BC2_SRGB
- 0x11: BC3_SRGB
- 0x12: BC7
- 0x13: BC7_SRGB


## About

*This is a format documentation originally made by Tyulis for the 3DSkit project, mostly based on the sources below.
It is not an absolute reference, and may contain wrong, outdated or incomplete stuff.
Sources used to make this document and contributors are listed below, the rest has been found by personal investigations.
If you find any error, incomplete or outdated stuff, dont't hesitate to open an issue or a pull request in the [3DSkit GitHub repository](https://github.com/Tyulis/3DSkit).
This document is completely free of charge, you can read it, use it, share it, modify it, sell it if you want without any conditions
(but leaving this paragraph and sharing extensions and corrections of this document on the original repository would just be the most basic of kindnesses)

Documentation about the structure of this document is [here](https://github.com/Tyulis/3DSkit/doc/README.md)*

## Credits and sources
- [3dstools](https://github.com/ObsidianX/3dstools), by ObsidianX
- [https://www.3dbrew.org/wiki/BCFNT]
- [https://avsys.xyz/wiki/BFFNT_(File_Format)]
