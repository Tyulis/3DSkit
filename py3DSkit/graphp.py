# -*- coding:utf-8 -*-

import numpy as np
from .graphx import *


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side='left')
    if idx > 0 and (idx == len(array) or np.fabs(value - np.int(array[idx - 1])) < np.fabs(value - np.int(array[idx]))):
        return idx - 1
    else:
        return idx


def _packBC4Texture(input, output, width, height, format, swizzlesize, littleendian):
    padwidth = 2 ** int(np.ceil(np.log2(width)))
    padheight = 2 ** int(np.ceil(np.log2(height)))
    twidth = padwidth // 4
    theight = padheight // 4
    swizzle = Swizzle(twidth, BC4_BYTESPERTEXEL, swizzlesize);
    texel = np.zeros((4, 4), dtype=np.uint8)
    indices = np.zeros((4, 4), dtype=np.uint8)
    for ytexel in range(theight):
        for xtexel in range(twidth):
            for ypix in range(4):
                for xpix in range(4):
                    ypos = ytexel * 4 + ypix
                    xpos = xtexel * 4 + xpix
                    if xpos >= width or ypos >= height:
                        texel[ypix, xpix] = 0
                    else:
                        position = (ypos * width + xpos) * 4
                        texel[ypix, xpix] = np.uint8(np.round(input[position] * 0.2989 + input[position + 1] * 0.5870 + input[position + 2] * 0.1140))
            # TODO : naive implementation, to improve
            maxvalue = np.max(texel)
            minvalue = np.min(texel)
            if maxvalue == minvalue:
                if maxvalue >= 0xFF:
                    minvalue -= 1
                else:
                    maxvalue += 1
            step = (maxvalue - minvalue) / 7
            slices = np.asarray([minvalue + step * i for i in range(8)], dtype=np.uint8)
            for ypix in range(4):
                for xpix in range(4):
                    index = find_nearest(slices, texel[ypix, xpix])
                    if index == 7:
                        index = 0
                    elif index == 0:
                        index = 1
                    else:
                        index = 8 - index
                    indices[ypix, xpix] = index
            if swizzlesize < 0:
                offset = (ytexel * twidth + xtexel) * BC4_BYTESPERTEXEL
            else:
                offset = swizzle.getSwizzleOffset(xtexel, ytexel)
            output[offset] = maxvalue
            output[offset + 1] = minvalue
            line1 = line2 = 0
            for i in range(8):
                line1 |= indices[i // 4][i % 4] << (i * 3)
            for i in range(8, 16):
                line2 |= indices[i // 4][i % 4] << ((i - 8) * 3)
            if littleendian:
                output[offset + 4] = line1 >> 16
                output[offset + 3] = (line1 >> 8) & 0xFF
                output[offset + 2] = (line1 & 0xFF)
                output[offset + 7] = line2 >> 16
                output[offset + 6] = (line2 >> 8) & 0xFF
                output[offset + 5] = (line2 & 0xFF)
            else:
                output[offset + 2] = line1 >> 16
                output[offset + 3] = (line1 >> 8) & 0xFF
                output[offset + 4] = (line1 & 0xFF)
                output[offset + 5] = line2 >> 16
                output[offset + 6] = (line2 >> 8) & 0xFF
                output[offset + 7] = (line2 & 0xFF)

def _packTiledTexture(input, output, width, height, format, swizzlesize, littleendian):
    tilesx = int(np.ceil(width / 8))
    tilesy = int(np.ceil(height / 8))
    datawidth = 1 << int(np.ceil(LOG2(width)))
    #int dataheight = 1 << (int)ceil(LOG2((double)height));
    totalx = int(np.ceil(datawidth / 8.0))
    #int totaly = (int)ceil((double)dataheight / 8.0);
    pxsize = getPixelSize(format)
    if format == L4 or format == A4:
        swizzle = Swizzle(width, 1, swizzlesize)
    else:
        swizzle = Swizzle(width, pxsize, swizzlesize)
    for ytile in range(tilesy):
        for xtile in range(tilesx):
            for ysub in range(2):
                for xsub in range(2):
                    for yblock in range(2):
                        for xblock in range(2):
                            for ypix in range(2):
                                for xpix in range(2):
                                    ypos = ytile * 8 + ysub * 4 + yblock * 2 + ypix
                                    xpos = xtile * 8 + xsub * 4 + xblock * 2 + xpix
                                    position = (ypos * width + xpos) * 4
                                    r, g, b, a = input[position: position + 4]
                                    if format in (L4, A4):
                                        shift = xpix * 4
                                        if swizzlesize < 0:
                                            outpos = ytile * totalx * 32 + xtile * 32 + ysub * 16 + xsub * 8 + yblock * 4 + xblock * 2 + ypix
                                        else:
                                            outpos = swizzle.getSwizzleOffset(xtile * 8 + xsub * 4 + xblock * 2, ytile * 8 + ysub * 4 + yblock * 2 + ypix)
                                        if format == L4:
                                            l = np.uint8(np.round(r * 0.2989 + g * 0.5870 + b * 0.1140)) // 16
                                            output[outpos] |= l << shift
                                        elif format == A4:
                                            output[outpos] |= (a // 16) << shift
                                    else:
                                        if swizzlesize < 0:
                                            outpos = (ytile * totalx * 64 + xtile * 64 + ysub * 32 + xsub * 16 + yblock * 8 + xblock * 4 + ypix * 2 + xpix) * pxsize
                                        else:
                                            outpos = swizzle.getSwizzleOffset(xpos, ypos)
                                        if littleendian:
                                            if format == RGBA8 or format == RGBA8_SRGB:
                                                output[outpos: outpos + 4] = r, g, b, a
                                            elif format == RGB8:
                                                output[outpos: outpos + 3] = b, g, r
                                            elif format == RGBA5551:
                                                rgba = (round(r / 8.225806451612) << 11) | (round(g / 8.225806451612) << 6) | (round(b / 8.225806451612) << 1) | (a > 127)
                                                output[outpos] = rgba & 0xFF
                                                output[outpos + 1] = rgba >> 8
                                            elif format == RGB565:
                                                rgb = (round(r / 8.225806451612) << 11) | (round(g / 4.0476190476190) << 5) | (round(b / 8.225806451612) << 1) | (a > 127)
                                                output[outpos] = rgb & 0xFF
                                                output[outpos + 1] = rgb >> 8
                                            elif format == RGBA4:
                                                outpos[outpos] = ((b // 16) << 4) | (a // 16)
                                                output[outpos + 1] = ((r // 16) << 4) | (g // 16)
                                            elif format == LA8:
                                                output[outpos] = a
                                                output[outpos + 1] = np.uint8(np.round(r * 0.2989 + g * 0.5870 + b * 0.1140))
                                            elif format == RG8:
                                                output[outpos] = g
                                                output[outpos] = r
                                            elif format == L8:
                                                output[outpos] = np.uint8(np.round(r * 0.2989 + g * 0.5870 + b * 0.1140))
                                            elif format == A8:
                                                output[outpos] = a
                                            elif format == LA4:
                                                l = np.uint8(np.round(r * 0.2989 + g * 0.5870 + b * 0.1140)) // 16
                                                output[outpos] = (l << 4) | (a // 16)
                                        else:
                                            if format == RGBA8 or format == RGBA8_SRGB:
                                                output[outpos: outpos + 4] = r, g, b, a
                                            elif format == RGB8:
                                                output[outpos: outpos + 3] = r, g, b
                                            elif format == RGBA5551:
                                                rgba = (round(r / 8.225806451612) << 11) | (round(g / 8.225806451612) << 6) | (round(b / 8.225806451612) << 1) | (a > 127)
                                                output[outpos] = rgba >> 8
                                                output[outpos + 1] = rgba & 0xFF
                                            elif format == RGB565:
                                                rgb = (round(r / 8.225806451612) << 11) | (round(g / 4.0476190476190) << 5) | (round(b / 8.225806451612) << 1) | (a > 127)
                                                output[outpos] = rgb >> 8
                                                output[outpos + 1] = rgb & 0xFF
                                            elif format == RGBA4:
                                                output[outpos] = ((r // 16) << 4) | (g // 16)
                                                outpos[outpos + 1] = ((b // 16) << 4) | (a // 16)
                                            elif format == LA8:
                                                output[outpos] = np.uint8(np.round(r * 0.2989 + g * 0.5870 + b * 0.1140))
                                                output[outpos + 1] = a
                                            elif format == RG8:
                                                output[outpos] = r
                                                output[outpos] = g
                                            elif format == L8:
                                                output[outpos] = np.uint8(np.round(r * 0.2989 + g * 0.5870 + b * 0.1140))
                                            elif format == A8:
                                                output[outpos] = a
                                            elif format == LA4:
                                                l = np.uint8(np.round(r * 0.2989 + g * 0.5870 + b * 0.1140)) // 16
                                                output[outpos] = (l << 4) | (a // 16)

def packTexture(indata, outdata, width, height, format, swizzle, littleendian):
    if format == BC4:
        _packBC4Texture(indata, outdata, width, height, format, swizzle, littleendian)
    else:
        _packTiledTexture(indata, outdata, width, height, format, swizzle, littleendian)
