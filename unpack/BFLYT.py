# -*- coding:utf-8 -*-
from util.txtree import dump
from collections import OrderedDict
from util import error
import util.rawutil as rawutil
from util.funcops import ClsFunc, byterepr
from util.fileops import write, make_outfile

FLYT_HEADER = '4s4HI2H'
WRAPS = (
	'Near-Clamp',
	'Near-Repeat',
	'Near-Mirror',
	'GX2-Mirror-Once',
	'Clamp',
	'Repeat',
	'Mirror',
	'GX2-Mirror-Once-Border'
)
MAPPING_METHODS = (
	'UV-Mapping',
	'',
	'',
	'Orthogonal-Projection',
	'PaneBasedProjection'
)
BLENDS = (
	'Max',
	'Min'
)

COLOR_BLENDS = (
	'Overwrite',
	'Multiply',
	'Add',
	'Exclude',
	'4',
	'Subtract',
	'Dodge',
	'Burn',
	'Overlay',
	'Indirect',
	'Blend-Indirect',
	'Each-Indirect'
)

ALPHA_COMPARE_CONDITIONS = (
	'Never',
	'Less',
	'Less-or-Equal',
	'Equal',
	'Not-Equal',
	'Greater-or-Equal',
	'Greater',
	'Always'
)
BLEND_CALC = (
	'0',
	'1',
	'FBColor',
	'1-FBColor',
	'PixelAlpha',
	'1-PixelAlpha',
	'FBAlpha',
	'1-FBAlpha',
	'PixelColor',
	'1-PixelColor'
)

BLEND_CALC_OPS = (
	'0',
	'Add',
	'Subtract',
	'Reverse-Subtract',
	'Min',
	'Max'
)

LOGICAL_CALC_OPS = (
	'None',
	'NoOp',
	'Clear',
	'Set',
	'Copy',
	'InvCopy',
	'Inv',
	'And',
	'Nand',
	'Or',
	'Nor',
	'Xor',
	'Equiv',
	'RevAnd',
	'InvAnd',
	'RevOr',
	'InvOr'
)

PROJECTION_MAPPING_TYPES = (
	'Standard',
	'Entire-Layout',
	'2',
	'3',
	'Pane-RandS-Projection',
	'5',
	'6'
)

TEXT_ALIGNS = (
	'NA',
	'Left',
	'Center',
	'Right'
)
ORIG_X = (
	'Center',
	'Left',
	'Right'
)
ORIG_Y = (
	'Center',
	'Up',
	'Down'
)


class extractBFLYT(ClsFunc, rawutil.TypeReader):
	def main(self, filename, data, verbose, opts={}):
		self.verbose = verbose
		outfile = make_outfile(filename, 'tflyt')
		self.bflyt = data
		self.readheader()
		write(dump(self.parsedata()), outfile)
	
	def color(self, data, ptr, format):
		format = format.upper().strip()
		if format == 'RGBA8':
			sz = 4
		elif format == 'RGB8':
			sz = 3
		if format in ('RGBA8', 'RGB8'):
			r = data[ptr]
			g = data[ptr + 1]
			b = data[ptr + 2]
			if format == 'RGBA8':
				a = data[ptr + 3]
			final = OrderedDict()
			final['RED'] = r
			final['GREEN'] = g
			final['BLUE'] = b
			if format == 'RGBA8':
				final['ALPHA'] = a
		return final, ptr + sz
	
	def readheader(self):
		header = self.bflyt[:0x14]
		self.data = self.bflyt[0x14:]
		if header[0:4] != b'FLYT':
			error('Invalid magic %s, expected FLYT' % byterepr(header[0:4]), 301)
		self.byteorder = '<' if header[4:6] == b'\xff\xfe' else '>'
		self.endianname = 'little' if self.byteorder == '<' else 'big'
		hdata = self.unpack(FLYT_HEADER, header)
		#some unused or already read data
		#magic, endianness,
		#headerlength = hdata[2]
		self.version = hdata[3]
		#padding = hdata[4]
		#filesize = hdata[5]
		self.secnum = hdata[6]
		#padding = hdata[7]
		if self.verbose:
			print('Version: %08x' % self.version)
			print('Section count: %d' % self.secnum)
	
	def parsedata(self):
		ptr = 0
		self.tree = OrderedDict()
		self.tree['byte-order'] = self.byteorder
		self.tree['version'] = self.version
		self.tree['BFLYT'] = OrderedDict()
		self.actnode = self.tree['BFLYT']  # creates a pointer in the tree, which can change later
		self.actnode['__pan1idx'] = 0
		self.actnode['__pas1idx'] = 0
		for i in range(0, self.secnum):
			magic = self.data[ptr:ptr + 4].decode('ascii')
			size = int.from_bytes(self.data[ptr + 4:ptr + 8], self.endianname)
			chunk = self.data[ptr:ptr + size]
			try:
				method = eval('self.read' + magic)  # quicker to code than if magic=='lyt1':...
			except AttributeError:
				print('Invalid section magic: %s' % magic)
			method(chunk)
			ptr += size
		return self.tree
	
	def readlyt1(self, data):
		self.actnode['lyt1'] = OrderedDict()
		localnode = self.actnode['lyt1']
		ptr = 8
		localnode['drawn-from-middle'] = bool(self.uint8(data, ptr)[0]); ptr += 4
		localnode['screen-width'], ptr = self.float32(data, ptr)
		localnode['screen-height'], ptr = self.float32(data, ptr)
		localnode['max-parts-width'], ptr = self.float32(data, ptr)
		localnode['max-parts-height'], ptr = self.float32(data, ptr)
		localnode['name'], ptr = self.string(data, ptr)
	
	def readtxl1(self, data):
		self.actnode['txl1'] = OrderedDict()
		localnode = self.actnode['txl1']
		ptr = 8
		texnum, ptr = self.uint16(data, ptr)
		localnode['texture-number'] = texnum
		#localnode['data-start-offset'] = self.uint16(data,ptr)
		ptr += 2
		offsets = []
		startentries = ptr
		for i in range(0, texnum):
			offsets.append(self.uint32(data, ptr)[0]); ptr += 4
		filenames = []
		for offset in offsets:
			absoffset = startentries + offset
			filenames.append(self.string(data, absoffset)[0])
		localnode['file-names'] = filenames
		self.texturenames = filenames
	
	def readfnl1(self, data):
		self.actnode['fnl1'] = OrderedDict()
		localnode = self.actnode['fnl1']
		ptr = 8
		fontnum, ptr = self.uint16(data, ptr)
		localnode['fonts-number'] = fontnum
		#localnode['data-start-offset'] = self.uint16(data, ptr)
		ptr += 2
		offsets = []
		startentries = ptr
		for i in range(0, fontnum):
			offsets.append(self.uint32(data, ptr)[0]); ptr += 4
		filenames = []
		for offset in offsets:
			absoffset = startentries + offset
			filenames.append(self.string(data, absoffset)[0])
		localnode['file-names'] = filenames
		self.fontnames = filenames
	
	def readmat1(self, data):
		self.actnode['mat1'] = OrderedDict()
		localnode = self.actnode['mat1']
		ptr = 8
		matnum, ptr = self.uint16(data, ptr)
		localnode['materials-number'] = matnum
		#localnode['data-start-offset'] = self.uint16(data, ptr)
		ptr += 2
		offsets = []
		for i in range(0, matnum):
			offsets.append(self.uint32(data, ptr)[0]); ptr += 4
		self.materials = []
		for offset in offsets:
			ptr = offset
			mat = OrderedDict()
			mat['name'] = self.string(data, ptr)[0]; ptr += 28
			mat['fore-color'], ptr = self.color(data, ptr, 'RGBA8')
			mat['back-color'], ptr = self.color(data, ptr, 'RGBA8')
			
			flags, ptr = self.uint32(data, ptr)
			if flags in (2069, 2154):  #to avoid many problems
				flags ^= 0x0800
				mat['false-0x800'] = True
			else:
				mat['false-0x800'] = False
			texref = flags & 0x00000003
			textureSRT = (flags & 0x0000000c) >> 2
			mappingSettings = (flags & 0x00000030) >> 4
			textureCombiners = (flags & 0x000000c0) >> 6
			blendMode = (flags & 0x00000300) >> 8
			blendAlpha = (flags & 0x00000c00) >> 10
			indirect = (flags & 0x00001000) >> 12
			alphaCompare = (flags & 0x00002000) >> 13
			projectionMapping = (flags & 0x0000c000) >> 14
			shadowBlending = (flags & 0x00030000) >> 16
			
			for i in range(0, texref):
				mat['texref-%d' % i] = OrderedDict()
				flagnode = mat['texref-%d' % i]
				flagnode['file'] = self.texturenames[self.uint16(data, ptr)[0]]; ptr += 2
				flagnode['wrap-S'] = WRAPS[self.uint8(data, ptr)[0]]; ptr += 1
				flagnode['wrap-T'] = WRAPS[self.uint8(data, ptr)[0]]; ptr += 1
			for i in range(0, textureSRT):
				mat['textureSRT-%d' % i] = OrderedDict()
				flagnode = mat['textureSRT-%d' % i]
				flagnode['X-translate'], ptr = self.float32(data, ptr)
				flagnode['Y-translate'], ptr = self.float32(data, ptr)
				flagnode['rotate'], ptr = self.float32(data, ptr)
				flagnode['X-scale'], ptr = self.float32(data, ptr)
				flagnode['Y-scale'], ptr = self.float32(data, ptr)
			for i in range(0, mappingSettings):
				mat['mapping-settings-%d' % i] = OrderedDict()
				flagnode = mat['mapping-settings-%d' % i]
				flagnode['unknown-1'], ptr = self.uint8(data, ptr)
				flagnode['mapping-method'] = MAPPING_METHODS[self.uint8(data, ptr)[0]]; ptr += 1
				flagnode['unknown-2'], ptr = self.uint8(data, ptr)
				flagnode['unknown-3'], ptr = self.uint8(data, ptr)
				flagnode['unknown-4'], ptr = self.uint8(data, ptr)
				flagnode['unknown-5'], ptr = self.uint8(data, ptr)
				flagnode['unknown-6'], ptr = self.uint8(data, ptr)
				flagnode['unknown-7'], ptr = self.uint8(data, ptr)
			
			for i in range(0, textureCombiners):
				mat['texture-combiner-%d' % i] = OrderedDict()
				flagnode = mat['texture-combiner-%d' % i]
				flagnode['color-blend'] = COLOR_BLENDS[self.uint8(data, ptr)[0]]; ptr += 1
				flagnode['alpha-blend'] = BLENDS[self.uint8(data, ptr)[0]]; ptr += 1
				flagnode['unknown-1'], ptr = self.uint8(data, ptr)
				flagnode['unknown-2'], ptr = self.uint8(data, ptr)
			if alphaCompare:
				mat['alpha-compare'] = OrderedDict()
				flagnode = mat['alpha-compare']
				flagnode['condition'] = ALPHA_COMPARE_CONDITIONS[self.uint8(data, ptr)[0]]; ptr += 1
				flagnode['unknown-1'], ptr = self.uint8(data, ptr)
				flagnode['unknown-2'], ptr = self.uint8(data, ptr)
				flagnode['unknown-3'], ptr = self.uint8(data, ptr)
				flagnode['value'], ptr = self.float32(data, ptr)
			for i in range(0, blendMode):
				mat['blend-mode-%d' % i] = OrderedDict()
				flagnode = mat['blend-mode-%d' % i]
				op = self.uint8(data, ptr)[0]
				flagnode['blend-operation'] = BLEND_CALC_OPS[op]; ptr += 1
				flagnode['source'] = BLEND_CALC[self.uint8(data, ptr)[0]]; ptr += 1
				flagnode['destination'] = BLEND_CALC[self.uint8(data, ptr)[0]]; ptr += 1
				flagnode['logical-operation'] = LOGICAL_CALC_OPS[self.uint8(data, ptr)[0]]; ptr += 1
			for i in range(0, blendAlpha):
				mat['blend-alpha-%d' % i] = OrderedDict()
				flagnode = mat['blend-alpha-%d' % i]
				flagnode['blend-operation'] = BLEND_CALC_OPS[self.uint8(data, ptr)[0]]; ptr += 1
				flagnode['source'] = BLEND_CALC[self.uint8(data, ptr)[0]]; ptr += 1
				flagnode['destination'] = BLEND_CALC[self.uint8(data, ptr)[0]]; ptr += 1
				flagnode['unknown'], ptr = self.uint8(data, ptr)
			if indirect:
				mat['indirect-adjustment'] = OrderedDict()
				flagnode = mat['indirect-adjustment']
				flagnode['rotate'], ptr = self.float32(data, ptr)
				flagnode['X-warp'], ptr = self.float32(data, ptr)
				flagnode['Y-warp'], ptr = self.float32(data, ptr)
			for i in range(0, projectionMapping):
				mat['projection-mapping-%d' % i] = OrderedDict()
				flagnode = mat['projection-mapping-%d' % i]
				flagnode['X-translate'], ptr = self.float32(data, ptr)
				flagnode['Y-translate'], ptr = self.float32(data, ptr)
				flagnode['X-scale'], ptr = self.float32(data, ptr)
				flagnode['Y-scale'], ptr = self.float32(data, ptr)
				opt = self.uint8(data, ptr)[0]
				flagnode['option'] = PROJECTION_MAPPING_TYPES[opt]; ptr += 1
				flagnode['unknown-1'], ptr = self.uint8(data, ptr)
				flagnode['unknown-2'], ptr = self.uint16(data, ptr)
			if shadowBlending:
				mat['shadow-blending'] = OrderedDict()
				flagnode = mat['shadow-blending']
				flagnode['black-blending'], ptr = self.color(data, ptr, 'RGB8')
				flagnode['white-blending'], ptr = self.color(data, ptr, 'RGBA8')
				pad, ptr = self.uint8(data, ptr)
			self.materials.append(mat)
		if ptr < len(data):
				extra = data[ptr:]
				localnode['extra'] = extra.hex()
		localnode['materials'] = self.materials
	
	def readpane(self, data, ptr):
		node = OrderedDict()
		flags, ptr = self.uint8(data, ptr)
		node['visible'] = bool(flags & 0b00000001)
		node['transmit-alpha-to-children'] = bool((flags & 0b00000010) >> 1)
		node['position-adjustment'] = bool((flags & 0b00000100) >> 2)
		origin, ptr = self.uint8(data, ptr)
		mainorigin = origin % 16
		parentorigin = origin // 16
		node['origin'] = OrderedDict()
		node['parent-origin'] = OrderedDict()
		orignode = node['origin']
		orignode['x'] = ORIG_X[mainorigin % 4]
		orignode['y'] = ORIG_Y[mainorigin // 4]
		orignode = node['parent-origin']
		orignode['x'] = ORIG_X[parentorigin % 4]
		orignode['y'] = ORIG_Y[parentorigin // 4]
		node['alpha'], ptr = self.uint8(data, ptr)
		node['part-scale'], ptr = self.uint8(data, ptr)
		node['name'] = self.string(data, ptr)[0]
		ptr += 32
		self.actnode['__prevname'] = node['name']
		node['X-translation'], ptr = self.float32(data, ptr)
		node['Y-translation'], ptr = self.float32(data, ptr)
		node['Z-translation'], ptr = self.float32(data, ptr)
		node['X-rotation'], ptr = self.float32(data, ptr)
		node['Y-rotation'], ptr = self.float32(data, ptr)
		node['Z-rotation'], ptr = self.float32(data, ptr)
		node['X-scale'], ptr = self.float32(data, ptr)
		node['Y-scale'], ptr = self.float32(data, ptr)
		node['width'], ptr = self.float32(data, ptr)
		node['height'], ptr = self.float32(data, ptr)
		return node, ptr
	
	def readpan1(self, data):
		ptr = 8
		info, ptr = self.readpane(data, ptr)
		secname = 'pan1-%s' % self.actnode['__prevname']
		self.actnode[secname] = OrderedDict()
		self.actnode[secname].update(info)
		
	def readpas1(self, data):
		secname = 'pas1-%s' % self.actnode['__prevname']
		self.actnode[secname] = OrderedDict()
		parentnode = self.actnode
		self.actnode = self.actnode[secname]
		self.actnode['__parentnode'] = parentnode
	
	def readwnd1(self, data):
		ptr = 8
		info, ptr = self.readpane(data, ptr)
		secname = 'wnd1-%s' % self.actnode['__prevname']
		self.actnode[secname] = OrderedDict()
		localnode = self.actnode[secname]
		localnode.update(info)
		localnode['stretch-left'], ptr = self.uint16(data, ptr)
		localnode['stretch-right'], ptr = self.uint16(data, ptr)
		localnode['stretch-up'], ptr = self.uint16(data, ptr)
		localnode['stretch-down'], ptr = self.uint16(data, ptr)
		localnode['custom-left'], ptr = self.uint16(data, ptr)
		localnode['custom-right'], ptr = self.uint16(data, ptr)
		localnode['custom-up'], ptr = self.uint16(data, ptr)
		localnode['custom-down'], ptr = self.uint16(data, ptr)
		framenum, ptr = self.uint8(data, ptr)
		localnode['frame-count'] = framenum
		localnode['flags'], ptr = self.uint8(data, ptr)
		pad, ptr = self.uint16(data, ptr)
		offset1, ptr = self.uint32(data, ptr)
		offset2, ptr = self.uint32(data, ptr)
		localnode['color-1'], ptr = self.color(data, ptr, 'RGBA8')
		localnode['color-2'], ptr = self.color(data, ptr, 'RGBA8')
		localnode['color-3'], ptr = self.color(data, ptr, 'RGBA8')
		localnode['color-4'], ptr = self.color(data, ptr, 'RGBA8')
		matnum, ptr = self.uint16(data, ptr)
		localnode['material'] = self.materials[matnum]['name']
		coordsnum, ptr = self.uint8(data, ptr)
		localnode['coordinates-count'] = coordsnum
		pad, ptr = self.uint8(data, ptr)
		for i in range(0, coordsnum):
			localnode['coords-%d' % i] = OrderedDict()
			coordnode = localnode['coords-%d' % i]
			for j in range(0, 8):
				coordnode['texcoord-%d' % j], ptr = self.float32(data, ptr)
		wnd4offsets = []
		for i in range(0, framenum):
			wnd4offsets.append(self.uint32(data, ptr)[0]); ptr += 4
		wndmat = []
		for i in range(0, framenum):
			dic = OrderedDict()
			dic['material'] = self.materials[self.uint16(data, ptr)[0]]['name']; ptr += 2
			dic['index'], ptr = self.uint8(data, ptr)
			wndmat.append(dic)
			pad, ptr = self.uint8(data, ptr)
		localnode['wnd4-materials'] = wndmat
	
	def readtxt1(self, data):
		ptr = 8
		info, ptr = self.readpane(data, ptr)
		secname = 'txt1-%s' % self.actnode['__prevname']
		self.actnode[secname] = OrderedDict()
		localnode = self.actnode[secname]
		localnode.update(info)
		localnode['restrict-length'], ptr = self.uint16(data, ptr)
		localnode['length'], ptr = self.uint16(data, ptr)
		localnode['material'] = self.materials[self.uint16(data, ptr)[0]]['name']; ptr += 2
		localnode['font'] = self.fontnames[self.uint16(data, ptr)[0]]; ptr += 2
		align, ptr = self.uint8(data, ptr)
		localnode['alignment'] = OrderedDict()
		localnode['alignment']['x'] = ORIG_X[align % 4]
		localnode['alignment']['y'] = ORIG_Y[align // 4]
		localnode['line-alignment'] = TEXT_ALIGNS[self.uint8(data, ptr)[0]]; ptr += 1
		localnode['active-shadows'], ptr = self.uint8(data, ptr)
		localnode['unknown-1'], ptr = self.uint8(data, ptr)
		localnode['italic-tilt'], ptr = self.float32(data, ptr)
		startoffset, ptr = self.uint32(data, ptr)
		localnode['top-color'], ptr = self.color(data, ptr, 'RGBA8')
		localnode['bottom-color'], ptr = self.color(data, ptr, 'RGBA8')
		localnode['font-size-x'], ptr = self.float32(data, ptr)
		localnode['font-size-y'], ptr = self.float32(data, ptr)
		localnode['char-space'], ptr = self.float32(data, ptr)
		localnode['line-space'], ptr = self.float32(data, ptr)
		callnameoffset, ptr = self.uint32(data, ptr)
		localnode['shadow'] = OrderedDict()
		shadownode = localnode['shadow']
		shadownode['offset-X'], ptr = self.float32(data, ptr)
		shadownode['offset-Y'], ptr = self.float32(data, ptr)
		shadownode['scale-X'], ptr = self.float32(data, ptr)
		shadownode['scale-Y'], ptr = self.float32(data, ptr)
		shadownode['top-color'], ptr = self.color(data, ptr, 'RGBA8')
		shadownode['bottom-color'], ptr = self.color(data, ptr, 'RGBA8')
		shadownode['unknown-2'], ptr = self.uint32(data, ptr)
		ptr += 4
		text = data[ptr:ptr + localnode['length']].replace(b'\x00\x00', b'')
		#localnode['text']=text.hex()
		localnode['text'] = text.decode('utf-16-%s' % self.endianname[0] + 'e')
		ptr += len(text)
		ptr += 4 - (ptr % 4)
		callname = self.string(data, ptr)[0]
		localnode['call-name'] = callname
	
	def readusd1(self, data):
		ptr = 8
		secname = 'usd1-%s' % self.actnode['__prevname']
		self.actnode[secname] = OrderedDict()
		localnode = self.actnode[secname]
		entrynum, ptr = self.uint16(data, ptr)
		localnode['entry-number'] = entrynum
		localnode['unknown'], ptr = self.uint16(data, ptr)
		entries = []
		for i in range(0, entrynum):
			entry = OrderedDict()
			entryoffset = ptr
			nameoffset, ptr = self.uint32(data, ptr)
			nameoffset += entryoffset
			dataoffset, ptr = self.uint32(data, ptr)
			dataoffset += entryoffset
			datanum, ptr = self.uint16(data, ptr)
			datatype, ptr = self.uint8(data, ptr)
			unknown, ptr = self.uint8(data, ptr)
			if datatype == 0:
				entrydata = data[dataoffset:dataoffset + datanum].decode('ascii')
			elif datatype == 1:
				entrydata = []
				for j in range(0, datanum):
					entrydata.append(self.int32(data, dataoffset + (j * 4))[0])
			elif datatype == 2:
				entrydata = []
				for j in range(0, datanum):
					entrydata.append(self.float32(data, dataoffset + (j * 4))[0])
			entry['name'] = self.string(data, nameoffset)[0]
			entry['data'] = entrydata
			entry['unknown'] = unknown
			entries.append(entry)
		localnode['entries'] = entries
	
	def readpae1(self, data):
		self.actnode = self.actnode['__parentnode']
		secname = 'pae1-%s' % self.actnode['__prevname']
		self.actnode[secname] = 'End of %s' % self.actnode['__prevname']
	
	def readbnd1(self, data):
		ptr = 8
		info, ptr = self.readpane(data, ptr)
		secname = 'bnd1-%s' % self.actnode['__prevname']
		self.actnode[secname] = OrderedDict()
		localnode = self.actnode[secname]
		localnode.update(info)
	
	def readpic1(self, data):
		ptr = 8
		info, ptr = self.readpane(data, ptr)
		secname = 'pic1-%s' % self.actnode['__prevname']
		self.actnode[secname] = OrderedDict()
		localnode = self.actnode[secname]
		localnode.update(info)
		localnode['top-left-vtx-color'], ptr = self.color(data, ptr, 'RGBA8')
		localnode['top-right-vtx-color'], ptr = self.color(data, ptr, 'RGBA8')
		localnode['bottom-left-vtx-color'], ptr = self.color(data, ptr, 'RGBA8')
		localnode['bottom-right-vtx-color'], ptr = self.color(data, ptr, 'RGBA8')
		localnode['material'] = self.materials[self.uint16(data, ptr)[0]]['name']; ptr += 2
		localnode['tex-coords-number'], ptr = self.uint8(data, ptr)
		pad, ptr = self.uint8(data, ptr)
		coords = []
		for i in range(0, localnode['tex-coords-number']):
			entry = OrderedDict()
			entry['top-left'] = {'s': self.float32(data, ptr)[0], 't': self.float32(data, ptr + 4)[0]}; ptr += 8
			entry['top-right'] = {'s': self.float32(data, ptr)[0], 't': self.float32(data, ptr + 4)[0]}; ptr += 8
			entry['bottom-left'] = {'s': self.float32(data, ptr)[0], 't': self.float32(data, ptr + 4)[0]}; ptr += 8
			entry['bottom-right'] = {'s': self.float32(data, ptr)[0], 't': self.float32(data, ptr + 4)[0]}; ptr += 8
			coords.append(entry)
		localnode['tex-coords'] = coords
	
	def readprt1(self, data):
		ptr = 8
		info, ptr = self.readpane(data, ptr)
		secname = 'prt1-%s' % self.actnode['__prevname']
		self.actnode[secname] = OrderedDict()
		localnode = self.actnode[secname]
		localnode.update(info)
		count, ptr = self.uint32(data, ptr)
		localnode['section-count'] = count
		localnode['section-scale-X'], ptr = self.float32(data, ptr)
		localnode['section-scale-Y'], ptr = self.float32(data, ptr)
		entryoffsets = []
		extraoffsets = []
		entries = []
		for i in range(count):
			entry = OrderedDict()
			entry['name'] = self.string(data, ptr)[0]
			ptr += 24
			entry['unknown'], ptr = self.uint8(data, ptr)
			entry['flags'], ptr = self.uint8(data, ptr)
			padding, ptr = self.uint16(data, ptr)
			entryoffset, ptr = self.uint32(data, ptr)
			ptr += 4  #padding?
			extraoffset, ptr = self.uint32(data, ptr)
			entryoffsets.append(entryoffset)
			extraoffsets.append(extraoffset)
			entries.append(entry)
		if len(entryoffsets) == 0:
			localnode['extra'] = data[ptr:]
		else:
			localnode['extra'] = data[ptr: entryoffsets[0]]
		for i in range(count):
			entry = entries[i]
			parentnode = self.actnode
			self.actnode = entry
			self.actnode['__parentnode'] = parentnode
			entryoffset = entryoffsets[i]
			extraoffset = extraoffsets[i]
			if entryoffset != 0:
				length, ptr = self.uint32(data, entryoffset + 4)
				entrydata = data[entryoffset: entryoffset + length]
				magic = entrydata[0:4].decode('ascii')
				try:
					method = eval('self.read' + magic)  # quicker to code than if magic=='txt1':...
				except AttributeError:
					error('Invalid section magic: %s' % magic, 303)
				method(entrydata)
			if extraoffset != 0:
				extra = data[extraoffset:extraoffset + 48].hex()
				#key = list(entry.keys())[-1]
				entry['extra'] = extra
			self.actnode = self.actnode['__parentnode']
		localnode['entries'] = entries
		if len(extraoffsets) == 0:
			if len(entryoffsets) == 0:
				end = ptr
			else:
				lastentry = max(entryoffsets)
				end = lastentry + self.uint16(data, lastentry + 4)[0]
		else:
			if sum(extraoffsets) != 0:  #no extras
				end = max(extraoffsets) + 48
			else:
				end = entryoffset + length
		if end < len(data) and len(entryoffsets) != 0:
			localnode['dump'] = data[end:]
	
	def readgrp1(self, data):
		ptr = 8
		name = self.string(data, ptr)[0]; ptr += 34
		secname = 'grp1-%s' % name
		self.actnode[secname] = OrderedDict()
		localnode = self.actnode[secname]
		localnode['name'] = name
		subnum, ptr = self.uint16(data, ptr)
		subs = []
		for i in range(0, subnum):
			subs.append(self.string(data, ptr)[0]); ptr += 24
		localnode['subs'] = subs
		
	def readgrs1(self, data):
		if '__grsnum' not in self.actnode.keys():
			self.actnode['__grsnum'] = 0
		secname = 'grs1-%d' % self.actnode['__grsnum']
		self.actnode['__grsnum'] += 1
		self.actnode[secname] = OrderedDict()
		parentnode = self.actnode
		self.actnode = self.actnode[secname]
		self.actnode['__parentnode'] = parentnode
	
	def readgre1(self, data):
		self.actnode = self.actnode['__parentnode']
		secname = 'gre1-%d' % (self.actnode['__grsnum'] - 1)
		self.actnode[secname] = 'End of grs1-%d' % (self.actnode['__grsnum'] - 1)
	
	def readcnt1(self, data):
		ptr = 8
		self.actnode['cnt1'] = OrderedDict()
		localnode = self.actnode['cnt1']
		offset1, ptr = self.uint32(data, ptr)
		offset2, ptr = self.uint32(data, ptr)
		partnum, ptr = self.uint16(data, ptr)
		animnum, ptr = self.uint16(data, ptr)
		offset3, ptr = self.uint32(data, ptr)
		offset4, ptr = self.uint32(data, ptr)
		localnode['part-number'] = partnum
		localnode['anim-number'] = animnum
		name = self.string(data, ptr)[0]
		ptr += (len(name) + (4 - (len(name) % 4))) * 2
		localnode['name'] = name
		if partnum != 0:
			ptr = offset2
			parts = []
			for i in range(0, partnum):
				parts.append(self.string(data, ptr)[0]); ptr += 24
			localnode['parts'] = parts
		if animnum != 0:
			startpos = ptr
			animpartnum, ptr = self.uint32(data, ptr)
			animname, ptr = self.string(data, ptr)
			ptr += 4 - (ptr % 4)
			offsets = []
			localnode['anim-part'] = OrderedDict()
			localnode['anim-part']['name'] = animname
			localnode['anim-part']['anim-part-number'] = animpartnum
			for i in range(0, animpartnum):
				offsets.append(self.uint32(data, ptr)[0]); ptr += 4
			anims = []
			for offset in offsets:
				ptr = startpos + offset
				anims.append(self.string(data, ptr)[0])
			ptr += len(anims[-1])
			ptr += 4 - (ptr % 4)
			localnode['anim-part']['anims'] = anims
		if ptr < len(data):
			dump = data[ptr:]
			localnode['dump'] = dump
