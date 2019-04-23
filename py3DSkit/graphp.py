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
            #print('-------------------- (%d, %d)' % (xtexel, ytexel))
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
            #print(slices)
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
            #print(indices)
            #outpos = ytexel * 8 * (padwidth // 8) + xtexel * 8
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

def packTexture(indata, outdata, width, height, format, swizzle, littleendian):
    if format == BC4:
        _packBC4Texture(indata, outdata, width, height, format, swizzle, littleendian)
    else:
        _packTiledTexture(indata, outdata, width, height, format, swizzle, littleendian)
