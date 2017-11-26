# -*- coding:utf-8 -*-
from util.txtree import load
from util import error, BOMS
import util.rawutil as rawutil
from util.funcops import ClsFunc
from util.fileops import read, bwrite

FLYT_HEADER = '%s4s4H I 2H'
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


class TypeWriter (rawutil.TypeWriter):
	def magiccount(self, data, magic):
		count = 0
		for key in data.keys():
			if '-'.join(key.split('-')[:-1]) == magic:
				count += 1
		return count

	def sechdr(self, data, name):
		return self.pack('4sI', name.encode('ascii'), len(data) + 8)


class packBFLYT(ClsFunc, TypeWriter):
	def main(self, filenames, outname, endian, verbose, opts={}):
		filename = filenames[0]
		self.verbose = verbose
		tree = load(read(filename))
		if list(tree.keys())[2] != 'BFLYT':
			error('This is not a converted BFLYT file', 203)
		self.version = tree['version']
		self.byteorder = endian
		self.sections = tree['BFLYT']
		self.final = self.repackdata()
		bwrite(self.final, outname)
	
	def string4(self, s):
		s = s.encode('utf-8')
		pad = 4 - (len(s) % 4 or 4)
		return s + bytes(pad)

	def repackdata(self):
		self.secnum = 0
		data = self.repacktree(self.sections, True)
		hdr = self.repackhdr(data)
		return hdr + data

	def repackhdr(self, data):
		final = b'FLYT'
		final += rawutil.pack('>H', BOMS[self.byteorder])
		final += self.uint16(0x14)
		final += self.uint16(self.version)
		final += self.uint16(0x0702)
		final += self.uint32(len(data) + 0x14)
		final += self.uint16(self.secnum)
		final += self.uint16(0)
		return final

	def repacktree(self, tree, top=False, safe=False):
		final = b''
		for section in tree.keys():
			magic = section.split('-')[0]
			try:
				method = eval('self.pack%s' % magic)
			except AttributeError:
				if not safe:
					error('Invalid section: %s' % magic, 302)
				else:
					continue
			if top:
				self.secnum += 1
			final += method(tree[section])
		return final

	def packlyt1(self, data):
		final = b''
		final += self.uint8(data['drawn-from-middle'])
		final += self.pad(3)
		final += self.float32(data['screen-width'])
		final += self.float32(data['screen-height'])
		final += self.float32(data['max-parts-width'])
		final += self.float32(data['max-parts-height'])
		final += self.string4(data['name'])
		hdr = self.sechdr(final, 'lyt1')
		return hdr + final

	def packtxl1(self, data):
		final = b''
		final += self.uint16(data['texture-number'])
		final += self.uint16(0)  #data offset. Seems to be always 0
		filetable = b''
		offsets = []
		self.textures = data['file-names']
		offset_tbl_length = len(self.textures) * 4
		for name in self.textures:
			offsets.append(len(filetable) + offset_tbl_length)
			filetable += self.string(name)
		offsettbl = b''
		for offset in offsets:
			offsettbl += self.uint32(offset)
		final += offsettbl
		final += filetable
		if len(final) % 4 != 0:
			final += self.pad(4 - (len(final) % 4))
		hdr = self.sechdr(final, 'txl1')
		return hdr + final

	def packfnl1(self, data):
		final = b''
		final += self.uint16(data['fonts-number'])
		final += self.uint16(0)  #data offset. Seems to be always 0
		filetable = b''
		offsets = []
		filenames = data['file-names']
		self.fontnames = filenames
		offset_tbl_length = len(filenames) * 4
		for name in filenames:
			offsets.append(len(filetable) + offset_tbl_length)
			filetable += self.string(name)
		offsettbl = b''
		for offset in offsets:
			offsettbl += self.uint32(offset)
		final += offsettbl
		final += filetable
		final += self.pad(4 - (len(final) % 4))
		hdr = self.sechdr(final, 'fnl1')
		return hdr + final

	def packmat1(self, data):
		final = b''
		final += self.uint16(data['materials-number'])
		final += self.uint16(0)
		self.materials = data['materials']
		self.matnames = [el['name'] for el in self.materials]
		offsets = []
		offset_tbl_length = 12 + len(self.materials) * 4
		matdata = b''
		for mat in self.materials:
			offsets.append(offset_tbl_length + len(matdata))
			matsec = b''
			matsec += self.string(mat['name'], 0x1C)
			matsec += self.color(mat['fore-color'], 'RGBA8')
			matsec += self.color(mat['back-color'], 'RGBA8')
			flags = 0
			flags |= self.magiccount(mat, 'texref')
			flags |= self.magiccount(mat, 'textureSRT') << 2
			flags |= self.magiccount(mat, 'mapping-settings') << 4
			flags |= self.magiccount(mat, 'texture-combiner') << 6
			flags |= ('alpha-compare' in mat.keys()) << 8
			flags |= self.magiccount(mat, 'blend-mode') << 9
			flags |= self.magiccount(mat, 'blend-alpha') << 11
			flags |= ('indirect-adjustment' in mat.keys()) << 13
			flags |= self.magiccount(mat, 'projection-mapping') << 14
			flags |= ('shadow-blending' in mat.keys())
			if mat['false-0x800']:
				flags |= 0x800
			matsec += self.uint32(flags)
			items = list(mat.keys())[3:]
			for item in items:
				itemtype = '-'.join(item.split('-')[:-1])
				dic = mat[item]
				if itemtype == 'texref':
					matsec += self.uint16(self.textures.index(dic['file']))
					matsec += self.uint8(WRAPS.index(dic['wrap-S']))
					matsec += self.uint8(WRAPS.index(dic['wrap-T']))
				elif itemtype == 'textureSRT':
					matsec += self.float32(dic['X-translate'])
					matsec += self.float32(dic['Y-translate'])
					matsec += self.float32(dic['rotate'])
					matsec += self.float32(dic['X-scale'])
					matsec += self.float32(dic['Y-scale'])
				elif itemtype == 'mapping-settings':
					matsec += self.uint8(dic['unknown-1'])
					matsec += self.uint8(MAPPING_METHODS.index(dic['mapping-method']))
					matsec += self.uint8(dic['unknown-2'])
					matsec += self.uint8(dic['unknown-3'])
					matsec += self.uint8(dic['unknown-4'])
					matsec += self.uint8(dic['unknown-5'])
					matsec += self.uint8(dic['unknown-6'])
					matsec += self.uint8(dic['unknown-7'])
				elif itemtype == 'texture-combiner':
					matsec += self.uint8(COLOR_BLENDS.index(dic['color-blend']))
					matsec += self.uint8(BLENDS.index(dic['alpha-blend']))
					matsec += self.uint8(dic['unknown-1'])
					matsec += self.uint8(dic['unknown-2'])
				elif itemtype == 'alpha-compare':
					matsec += self.uint8(ALPHA_COMPARE_CONDITIONS.index(dic['condition']))
					matsec += self.uint8(dic['unknown-1'])
					matsec += self.uint8(dic['unknown-2'])
					matsec += self.uint8(dic['unknown-3'])
					matsec += self.uint32(dic['value'])
				elif itemtype == 'blend-mode':
					matsec += self.uint8(BLEND_CALC_OPS.index(dic['blend-operation']))
					matsec += self.uint8(BLEND_CALC.index(dic['source']))
					matsec += self.uint8(BLEND_CALC.index(dic['destination']))
					matsec += self.uint8(LOGICAL_CALC_OPS.index(dic['logical-operation']))
				elif itemtype == 'blend-alpha':
					matsec += self.uint8(BLEND_CALC_OPS.index(dic['blend-operation']))
					matsec += self.uint8(BLEND_CALC.index(dic['source']))
					matsec += self.uint8(BLEND_CALC.index(dic['destination']))
					matsec += self.uint8(dic['unknown'])
				elif itemtype == 'indirect-adjustment':
					matsec += self.float32(dic['rotate'])
					matsec += self.float32(dic['X-warp'])
					matsec += self.float32(dic['Y-warp'])
				elif itemtype == 'projection-mapping':
					matsec += self.float32(dic['X-translate'])
					matsec += self.float32(dic['Y-translate'])
					matsec += self.float32(dic['X-scale'])
					matsec += self.float32(dic['Y-scale'])
					matsec += self.uint8(PROJECTION_MAPPING_TYPES.index(dic['option']))
					matsec += self.uint8(dic['unknown-1'])
					matsec += self.uint16(dic['unknown-2'])
				elif itemtype == 'shadow-blending':
					matsec += self.color(dic['black-blending'], 'RGB8')
					matsec += self.color(dic['white-blending'], 'RGBA8')
					matsec += self.pad(1)
			matdata += matsec
		offsettbl = b''
		for offset in offsets:
			offsettbl += self.uint32(offset)
		final += offsettbl
		final += matdata
		if 'extra' in data.keys():
			final += bytes.fromhex(data['extra'])
		hdr = self.sechdr(final, 'mat1')
		return hdr + final

	def packpane(self, data):  #pane section: 76B
		panesec = b''
		flags = 0
		flags |= data['visible']
		flags |= data['transmit-alpha-to-children'] << 1
		flags |= data['position-adjustment'] << 2
		panesec += self.uint8(flags)
		origin_x = ORIG_X.index(data['origin']['x'])
		origin_y = ORIG_Y.index(data['origin']['y'])
		parent_origin_x = ORIG_X.index(data['parent-origin']['x'])
		parent_origin_y = ORIG_Y.index(data['parent-origin']['y'])
		main_origin = (origin_y * 4) + origin_x
		parent_origin = (parent_origin_y * 4) + parent_origin_x
		origin = (parent_origin * 16) + main_origin
		panesec += self.uint8(origin)
		panesec += self.uint8(data['alpha'])
		panesec += self.uint8(data['part-scale'])
		panesec += self.string(data['name'], 32)
		panesec += self.float32(data['X-translation'])
		panesec += self.float32(data['Y-translation'])
		panesec += self.float32(data['Z-translation'])
		panesec += self.float32(data['X-rotation'])
		panesec += self.float32(data['Y-rotation'])
		panesec += self.float32(data['Z-rotation'])
		panesec += self.float32(data['X-scale'])
		panesec += self.float32(data['Y-scale'])
		panesec += self.float32(data['width'])
		panesec += self.float32(data['height'])
		return panesec

	def packpan1(self, data):
		final = self.packpane(data)
		hdr = self.sechdr(final, 'pan1')
		return hdr + final

	def packpas1(self, data):
		tree = self.repacktree(data, True)
		pas1 = self.sechdr(b'', 'pas1')
		return pas1 + tree

	def packpae1(self, data):
		return self.sechdr(b'', 'pae1')

	def packwnd1(self, data):
		final = self.packpane(data)
		final += self.uint16(data['stretch-left'])
		final += self.uint16(data['stretch-right'])
		final += self.uint16(data['stretch-up'])
		final += self.uint16(data['stretch-down'])
		final += self.uint16(data['custom-left'])
		final += self.uint16(data['custom-right'])
		final += self.uint16(data['custom-up'])
		final += self.uint16(data['custom-down'])
		final += self.uint8(data['frame-count'])
		final += self.uint8(data['flags'])
		final += self.pad(2)
		final += self.uint32(0x70)  #the offset1. Always 0x70
		final += self.uint32(132 + (32 * data['coordinates-count']))  #the offset2
		final += self.color(data['color-1'], 'RGBA8')
		final += self.color(data['color-2'], 'RGBA8')
		final += self.color(data['color-3'], 'RGBA8')
		final += self.color(data['color-4'], 'RGBA8')
		final += self.uint16(self.matnames.index(data['material']))
		final += self.uint8(data['coordinates-count'])
		final += self.pad(1)
		for i in range(0, data['coordinates-count']):
			dic = data['coords-%d' % i]
			for texcoord in dic.values():
				final += self.float32(texcoord)
		part1len = len(final)
		for i in range(0, len(data['wnd4-materials'])):
			offset = part1len + 4 * (len(data['wnd4-materials'])) + (4 * i) + 8
			final += self.uint32(offset)
		for mat in data['wnd4-materials']:
			final += self.uint16(self.matnames.index(mat['material']))
			final += self.uint8(mat['index'])
			final += self.pad(1)
		hdr = self.sechdr(final, 'wnd1')
		return hdr + final

	def packtxt1(self, data):
		final = self.packpane(data)
		final += self.uint16(data['restrict-length'])
		final += self.uint16(data['length'])
		final += self.uint16(self.matnames.index(data['material']))
		final += self.uint16(self.fontnames.index(data['font']))
		align = (ORIG_Y.index(data['alignment']['y']) * 4) + ORIG_X.index(data['alignment']['x'])
		final += self.uint8(align)
		final += self.uint8(TEXT_ALIGNS.index(data['line-alignment']))
		final += self.uint8(data['active-shadows'])
		final += self.uint8(data['unknown-1'])
		final += self.float32(data['italic-tilt'])
		final += self.uint32(164)  #the start offset. Always 164
		final += self.color(data['top-color'], 'RGBA8')
		final += self.color(data['bottom-color'], 'RGBA8')
		final += self.float32(data['font-size-x'])
		final += self.float32(data['font-size-y'])
		final += self.float32(data['char-space'])
		final += self.float32(data['line-space'])
		final += self.uint32(0)
		shadow = data['shadow']
		final += self.float32(shadow['offset-X'])
		final += self.float32(shadow['offset-Y'])
		final += self.float32(shadow['scale-X'])
		final += self.float32(shadow['scale-Y'])
		final += self.color(shadow['top-color'], 'RGBA8')
		final += self.color(shadow['bottom-color'], 'RGBA8')
		final += self.uint32(shadow['unknown-2'])
		text = data['text'].encode('utf-16-%se' % ('l' if self.byteorder == '<' else 'b'))
		final += text
		if len(final) % 4 != 0:
			final += self.pad(4 - (len(text) % 4))
		final += data['call-name'].encode('ascii')  #because of padding issues
		if len(final) % 4 != 0:
			final += self.pad(4 - (len(text) % 4))
		hdr = self.sechdr(final, 'txt1')
		return hdr + final

	def packusd1(self, data):
		final = b''
		entrynum = data['entry-number']
		final += self.uint16(data['entry-number'])
		final += self.uint16(data['unknown'])
		nametbl = b''
		datatbl = b''
		nameoffsets = []
		dataoffsets = []
		for entry in data['entries']:
			nameoffsets.append(len(nametbl))
			nametbl += self.string(entry['name'])
			dataoffsets.append(len(datatbl))
			typename = entry['data'][0].__class__.__qualname__
			if typename == 'float':
				datatype = 2
			elif typename == 'int':
				datatype = 1
			elif typename in ('str', 'unicode', 'bytes'):  #...
				datatype = 0
			for el in entry['data']:
				if datatype == 0:
					datatbl += self.string(el)
				elif datatype == 1:
					datatbl += self.int32(el)
				elif datatype == 2:
					datatbl += self.float32(el)
		if len(datatbl) % 4 != 0:
			datatbl += self.pad(4 - (len(datatbl) % 4))
		if len(nametbl) % 4 != 0:
			nametbl += self.pad(4 - (len(nametbl) % 4))
		i = 0
		#entryoffset = len(final)
		for entry in data['entries']:  #1 entry in the table = 12B
			entryrest = entrynum - (i + 1)
			final += self.uint32((12 * entryrest) + len(datatbl) + nameoffsets[i] + 0x0c)
			final += self.uint32((12 * entryrest) + dataoffsets[i] + 0x0c)
			final += self.uint16(len(entry['data']))
			typename = entry['data'][0].__class__.__qualname__
			if typename == 'float':
				datatype = 2
			elif typename == 'int':
				datatype = 1
			elif typename in ('str', 'unicode', 'bytes'):  #...
				datatype = 0
			final += self.uint8(datatype)
			final += self.uint8(entry['unknown'])
		final += datatbl
		final += nametbl
		hdr = self.sechdr(final, 'usd1')
		return hdr + final

	def packpic1(self, data):
		final = self.packpane(data)
		final += self.color(data['top-left-vtx-color'], 'RGBA8')
		final += self.color(data['top-right-vtx-color'], 'RGBA8')
		final += self.color(data['bottom-left-vtx-color'], 'RGBA8')
		final += self.color(data['bottom-right-vtx-color'], 'RGBA8')
		final += self.uint16(self.matnames.index(data['material']))
		texcoordnum = data['tex-coords-number']
		final += self.uint8(texcoordnum)
		final += self.pad(1)
		for texcoord in data['tex-coords']:
			final += self.float32(texcoord['top-left']['s'])
			final += self.float32(texcoord['top-left']['t'])
			final += self.float32(texcoord['top-right']['s'])
			final += self.float32(texcoord['top-right']['t'])
			final += self.float32(texcoord['bottom-left']['s'])
			final += self.float32(texcoord['bottom-left']['t'])
			final += self.float32(texcoord['bottom-right']['s'])
			final += self.float32(texcoord['bottom-right']['t'])
		hdr = self.sechdr(final, 'pic1')
		return hdr + final

	def packbnd1(self, data):
		final = self.packpane(data)
		hdr = self.sechdr(final, 'bnd1')
		return hdr + final

	def packprt1(self, data):
		final = self.packpane(data)
		final += self.uint32(data['section-count'])
		final += self.float32(data['section-scale-X'])
		final += self.float32(data['section-scale-Y'])
		entrydata = b''
		extradata = b''
		dataoffsets = []
		extraoffsets = []
		for entry in data['entries']:
			sec = self.repacktree(entry, safe=True)
			if sec != b'':
				dataoffsets.append(len(entrydata) + len(data['entries']) * 36 + 88)
			else:
				dataoffsets.append(0)
			entrydata += sec
		for entry in data['entries']:
			if 'extra' in entry.keys():
				extraoffsets.append(len(extradata) + len(entrydata) + len(data['entries']) * 36 + 88)
				extradata += bytes.fromhex(entry['extra'])
			else:
				extraoffsets.append(0)
		i = 0
		for entry in data['entries']:  #1 entry=36B
			final += self.string(entry['name'], 24)
			final += self.uint8(entry['unknown-1'])
			final += self.uint8(entry['flags'])
			final += self.pad(2)
			final += self.uint32(dataoffsets[i])
			final += self.uint32(extraoffsets[i])
			i += 1
		if len(entrydata) % 4 != 0:
			entrydata += self.pad(4 - (len(entrydata) % 4))
		extradata += (self.pad(4 - (len(extradata) % 4)) if len(extradata) % 4 != 0 else b'')
		final += entrydata
		final += extradata
		if 'dump' in data.keys():
			final += data['dump']
		hdr = self.sechdr(final, 'prt1')
		return hdr + final

	def packgrp1(self, data):
		final = b''
		final += self.string(data['name'], 34)
		final += self.uint16(len(data['subs']))
		for sub in data['subs']:
			final += self.string(sub, 24)
		hdr = self.sechdr(final, 'grp1')
		return hdr + final

	def packgrs1(self, data):
		final = b''
		hdr = self.sechdr(final, 'grs1')
		content = self.repacktree(data, top=True)
		final += content
		return hdr + final

	def packgre1(self, data):
		final = b''
		hdr = self.sechdr(final, 'gre1')
		return hdr + final

	def packcnt1(self, data):
		final = b''
		sec1 = b''
		sec2 = b''
		sec3 = b''
		partnum = 0
		animnum = 0
		if 'parts' in data.keys():
			partnum = len(data['parts'])
			for part in data['parts']:
				sec1 += self.string(part, 24)
		if 'anim-part' in data.keys():
			animnode = data['anim-part']
			animnum = len(animnode['anims'])
			animname = self.string4(animnode['name'])
			sec2 += self.uint32(len(animname))
			sec2 += animname
			offsets = [4 * len(animnode['anims'])]
			names = self.string(animnode['anims'][0])
			for anim in animnode['anims'][1:]:
				offsets.append(offsets[0] + len(names))
				names += self.string(anim)
			names += self.pad(4 - (len(names) % 4))
			for offset in offsets:
				sec3 += self.uint32(offset)
			sec3 += names
		name = self.string(data['name'])
		if len(name) % 4 != 0:
			name += self.pad(4 - len(name) % 4)
		offset1 = len(name) + 28
		offset2 = len(name) * 2 + 28
		offset3 = len(sec1) + len(sec2) + len(name) * 2 + 28
		offset4 = offset3 + len(sec3)
		final += self.uint32(offset1)
		final += self.uint32(offset2)
		final += self.uint16(partnum)
		final += self.uint16(animnum)
		final += self.uint32(offset3)
		final += self.uint32(offset4)
		final += name + name
		final += sec1
		final += sec2
		final += sec3
		final += sec2  #?
		if 'dump' in data.keys():
			final += data['dump']
		hdr = self.sechdr(final, 'cnt1')
		return hdr + final
