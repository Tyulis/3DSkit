# -*- coding:utf-8 -*-
import os
import json
from collections import OrderedDict
from util import error, ENDIANS
from util.utils import byterepr, ClsFunc
from util.filesystem import *
import util.rawutil as rawutil

WRAPS = (
	'near clamp', 'near repeat', 'near mirror',
	'gx2 mirror once',
	'clamp', 'repeat', 'mirror',
	'gx2 mirror once border'
)

MAPPING_METHODS = ('uv mapping', '<unknown-1>', '<unknown-2>', 'orthogonal projection', 'pane based projection')
ALPHA_BLENDS = ('max', 'min', '<unknown-2>', '<unknown-3>', '<unknown-4>')

COLOR_BLENDS = (
	'overwrite', 'multiply', 'add', 'exclude',
	'<unknown-4>', 'subtract', 'dodge', 'burn',
	'overlay', 'indirect', 'blend indirect', 'each indirect'
)

ALPHA_COMPARE_CONDITIONS = (
	'never', 'less', 'less or equal',
	'equal', 'not equal',
	'greater or equal', 'greater', 'always'
)

BLEND_CALC = (
	'<unknown-0>', '<unknown-1>', 'fb color', 'fb color 1',
	'pixel alpha', 'pixel alpha 1', 'fb alpha', 'fb alpha 1',
	'pixel color', 'pixel color 1'
)

BLEND_OPS = (
	'<unknown-0>', 'add', 'subtract',
	'reverse subtract', 'min', 'max'
)

LOGICAL_OPS = (
	'none', 'no operation', 'clear', 'set',
	'copy', 'invert copy', 'invert', 'and',
	'not and', 'or', 'not or', 'xor',
	'equivalent', 'reverse and', 'invert and',
	'reverse or', 'invert or'
)

PROJECTION_MAPPING_TYPES = (
	'standard', 'entire layout', '<unknown-2>',
	'<unknown-3>', 'pane r and s projection', '<unknown-5>', '<unknown-6>'
)

TEXT_ALIGNS = ('undefined', 'left', 'center', 'right')
ORIG_X = ('center', 'left', 'right')
ORIG_Y = ('center', 'up', 'down')

class extractBFLYT (rawutil.TypeReader, ClsFunc):
	def main(self, filename, data, verbose, opts={}):
		self.outfile = make_outfile(filename, 'json')
		self.verbose = verbose
		self.output = OrderedDict()
		self.readheader(data)
		self.readdata(data)
		self.cleanout(self.output)
		with open(self.outfile, 'w') as f:
			json.dump(self.output, f, indent=4)

	def makenode(self, parent, name):
		parent[name] = OrderedDict()
		return parent[name]

	def cleanout(self, dic):
		toremove = []
		for key in dic.keys():
			if key.startswith('__'):
				toremove.append(key)
			elif isinstance(dic[key], OrderedDict):
				self.cleanout(dic[key])
		for key in toremove:
			del dic[key]

	def readheader(self, data):
		magic, bom = self.unpack_from('>4sH', data)
		if magic != b'FLYT':
			error.InvalidMagicError('Invalid magic %s, expected FLYT' % byterepr(magic))
		node = self.makenode(self.output, "FLYT")
		self.byteorder = ENDIANS[bom]
		headerlen, self.version, filesize, self.secnum, padding = self.unpack_from("H2I2H", data)
		node['byte order'] = self.byteorder
		node['version'] = self.version
		node['readable version'] = '%d.%d.%d.%d' % (self.version >> 24, (self.version >> 16) & 0xff, (self.version >> 8) & 0xFF, self.version & 0xFF)
		node['number of sections'] = self.secnum
		if self.verbose:
			print('Format version : %s' % node['readable version'])
			print('Number of sections : %d' % self.secnum)

	def readdata(self, data):
		self.curparent = self.output
		for i in range(self.secnum):
			startpos = data.tell()
			magic, size = self.unpack_from('4sI', data)
			data.seek(startpos)
			if magic in (b'pae1', b'gre1'):
				self.curparent = self.curparent['__parent']
			name, node = self.readsection(data, magic, startpos)
			node['__parent'] = self.curparent
			self.curparent[name] = node
			if magic in (b'pas1', b'grs1'):
				self.curparent = node
			data.seek(startpos + size)

	def readsection(self, data, magic, startpos):
		if magic == b'lyt1':
			name, node = self.readlyt1(data, startpos)
		elif magic == b'txl1':
			name, node = self.readtxl1(data, startpos)
		elif magic == b'fnl1':
			name, node = self.readfnl1(data, startpos)
		elif magic == b'mat1':
			name, node = self.readmat1(data, startpos)
		elif magic == b'pan1':
			name, node = self.readpan1(data, startpos)
		elif magic == b'pas1':
			name, node = self.readpas1(data, startpos)
		elif magic == b'pae1':
			name, node = self.readpae1(data, startpos)
		elif magic == b'bnd1':
			name, node = self.readbnd1(data, startpos)
		elif magic == b'wnd1':
			name, node = self.readwnd1(data, startpos)
		elif magic == b'pic1':
			name, node = self.readpic1(data, startpos)
		elif magic == b'txt1':
			name, node = self.readtxt1(data, startpos)
		elif magic == b'prt1':
			name, node = self.readprt1(data, startpos)
		elif magic == b'grp1':
			name, node = self.readgrp1(data, startpos)
		elif magic == b'grs1':
			name, node = self.readgrs1(data, startpos)
		elif magic == b'gre1':
			name, node = self.readgre1(data, startpos)
		elif magic == b'cnt1':
			name, node = self.readcnt1(data, startpos)
		elif magic == b'usd1':
			name, node = self.readusd1(data, startpos)
		else:
			error.InvalidSectionError('Invalid section magic %s at 0x%08X' % (byterepr(magic), startpos))
		return name, node

	def readlyt1(self, data, startpos):
		node = OrderedDict()
		magic, size, node['drawn from middle'], node['screen width'], node['screen height'], node['max parts width'], node['max parts height'], name = self.unpack_from('4sI ?3x4fn', data, startpos)
		node['name'] = name.decode('utf-8')
		return 'lyt1', node

	def readtxl1(self, data, startpos):
		node = OrderedDict()
		magic, size, texnum, offsets = self.unpack_from('4sI I/p1(I)', data)
		self.texnames = []
		for offset in offsets:
			string = self.unpack_from('n', data, startpos + 12 + offset)[0]
			self.texnames.append(string.decode('utf-8'))
		node['textures'] = self.texnames
		return 'txl1', node

	def readfnl1(self, data, startpos):
		node = OrderedDict()
		magic, size, fontnum, offsets = self.unpack_from('4sI I/p1(I)', data)
		self.fontnames = []
		for offset in offsets:
			string = self.unpack_from('n', data, startpos + 12 + offset)[0]
			self.fontnames.append(string.decode('utf-8'))
		node['fonts'] = self.fontnames
		return 'fnl1', node

	def readmat1(self, data, startpos):
		node = OrderedDict()
		magic, size, matnum, offsets = self.unpack_from('4sI I/p1(I)', data)
		self.materials = []
		self.matnames = []
		for offset in offsets:
			matnode = OrderedDict()
			if self.version < 0x08000000:
				name, forecolor, backcolor, flags = self.unpack_from('n28a (4B)(4B) I', data, startpos + offset)
				unknown = None
			else:
				name, flags, unknown, forecolor, backcolor = self.unpack_from('n28a I(4B)(4B)(4B)', data, startpos + offset)
			matnode['name'] = name.decode('utf-8')
			matnode['foreground color'] = forecolor
			matnode['background color'] = backcolor
			if unknown is not None:
				matnode['unknown'] = unknown
			if flags == 0x00000815:
				matnode['bad 0x800'] = True
				flags &= 0xFFFFF7FF
			else:
				matnode['bad 0x800'] = False
			texrefs = flags & 0x00000003
			textransforms = (flags & 0x0000000c) >> 2
			mapsettings = (flags & 0x00000030) >> 4
			texcombiners = (flags & 0x000000c0) >> 6
			alphacompare = (flags & 0x00000200) >> 9
			blendmodes = (flags & 0x00000c00) >> 10
			blendalpha = (flags & 0x00003000) >> 12
			indirect = (flags & 0x00004000) >> 14
			projectionmappings = (flags & 0x00018000) >> 15
			shadowblending = (flags & 0x00020000) >> 17
			for i in range(texrefs):  # 4B
				flagnode = self.makenode(matnode, 'texture reference %d' % i)
				index, wraps, wrapt = self.unpack_from('H2B', data)
				flagnode['texture'] = self.texnames[index]
				flagnode['wrap s'] = WRAPS[wraps]
				flagnode['wrap t'] = WRAPS[wrapt]
			for i in range(textransforms):  # 20B
				flagnode = self.makenode(matnode, 'texture transformation %d' % i)
				flagnode['x translation'], flagnode['y translation'], flagnode['rotation'], flagnode['x scale'], flagnode['y scale'] = self.unpack_from('5f', data)
			for i in range(mapsettings):  # 8B
				flagnode = self.makenode(matnode, 'mapping setting %d' % i)
				if self.version < 0x08000000:
					unk1, method, unk2 = self.unpack_from('2B(6B)', data)
				else:
					unk1, method, unk2 = self.unpack_from('2B(14B)', data)
				flagnode['method'] = MAPPING_METHODS[method]
				flagnode['unk1'] = unk1
				flagnode['unk2'] = unk2
			for i in range(texcombiners):  # 4B
				flagnode = self.makenode(matnode, 'texture combiner %d' % i)
				colorblend, alphablend, unk1, unk2 = self.unpack_from('4B', data)
				flagnode['color blending'] = COLOR_BLENDS[colorblend]
				flagnode['alpha blending'] = ALPHA_BLENDS[alphablend]
				flagnode['unk1'] = unk1
				flagnode['unk2'] = unk2
			if alphacompare:  # 8B
				flagnode = self.makenode(matnode, 'alpha comparison')
				condition, unk1, unk2, unk3, value = self.unpack_from('4Bf', data)
				flagnode['condition'] = ALPHA_COMPARE_CONDITIONS[condition]
				flagnode['value'] = value
				flagnode['unk1'] = unk1
				flagnode['unk2'] = unk2
				flagnode['unk3'] = unk3
			for i in range(blendmodes):  # 4B
				flagnode = self.makenode(matnode, 'blending mode %d' % i)
				operation, sourceblending, destblending, logical = self.unpack_from('4B', data)
				flagnode['operation'] = BLEND_OPS[operation]
				flagnode['source blending'] = BLEND_CALC[sourceblending]
				flagnode['destination blending'] = BLEND_CALC[destblending]
				flagnode['logical operation'] = LOGICAL_OPS[logical]
			for i in range(blendalpha):  # 4B
				flagnode = self.makenode(matnode, 'alpha blending mode %d' % i)
				operation, sourceblending, destblending, logical = self.unpack_from('4B', data)
				flagnode['operation'] = BLEND_OPS[operation]
				flagnode['source blending'] = BLEND_CALC[sourceblending]
				flagnode['destination blending'] = BLEND_CALC[destblending]
				flagnode['logical operation'] = LOGICAL_OPS[logical]
			if indirect:  # 12B
				flagnode = self.makenode(matnode, 'indirect adjustment')
				flagnode['rotation'], flagnode['warp x'], flagnode['warp y'] = self.unpack_from('3f', data)
			for i in range(projectionmappings):  # 20B
				flagnode = self.makenode(matnode, 'projection mapping %d' % i)
				flagnode['x translation'], flagnode['y translation'], flagnode['x scale'], flagnode['y scale'], option, unk1, unk2, unk3 = self.unpack_from('4f4B', data)
				flagnode['option'] = PROJECTION_MAPPING_TYPES[option]
				flagnode['unk1'] = unk1
				flagnode['unk2'] = unk2
				flagnode['unk3'] = unk3
			if shadowblending:  # 8B
				flagnode = self.makenode(matnode, 'shadow blending')
				flagnode['black blending'], flagnode['white blending'] = self.unpack_from('(3B)(4B)x', data)

			self.matnames.append(matnode['name'])
			self.materials.append(matnode)
		node['materials'] = self.materials
		return 'mat1', node

	def readpane(self, data, node):
		flags, origin, alpha, scale, name = self.unpack_from('4B |n32a', data)
		node['name'] = name.decode('utf-8')
		node['visible'] = bool(flags & 0b00000001)
		node['transmit alpha'] = bool(flags & 0b00000010)
		node['position adjustment'] = bool(flags & 0b00000100)
		node['x origin'] = ORIG_X[origin & 0b00000011]
		node['y origin'] = ORIG_Y[(origin & 0b00001100) >> 2]
		node['parent x origin'] = ORIG_X[(origin & 0b00110000) >> 4]
		node['parent y origin'] = ORIG_Y[(origin & 0b11000000) >> 6]
		node['alpha'] = alpha
		node['scale'] = scale
		node['x translation'], node['y translation'], node['z translation'], node['x rotation'], node['y rotation'], node['z rotation'], node['x scale'], node['y scale'], node['width'], node['height'] = self.unpack_from('10f', data)
		self.curparent['__lastpane'] = node['name']

	def readpan1(self, data, startpos):
		node = OrderedDict()
		magic, size = self.unpack_from('4sI', data)
		self.readpane(data, node)
		return 'pan1 - ' + node['name'], node

	def readpas1(self, data, startpos):
		node = OrderedDict()
		return 'pas1 - ' + self.curparent['__lastpane'], node

	def readpae1(self, data, startpos):
		node = OrderedDict()
		return 'pae1 - ' + self.curparent['__lastpane'], node

	def readbnd1(self, data, startpos):
		node = OrderedDict()
		magic, size = self.unpack_from('4sI', data)
		self.readpane(data, node)
		return 'bnd1 - ' + node['name'], node

	def readwnd1(self, data, startpos):
		node = OrderedDict()
		magic, size = self.unpack_from('4sI', data)
		self.readpane(data, node)
		node['left stretch'], node['right stretch'], node['top stretch'], node['bottom stretch'] = self.unpack_from('4H', data)
		node['custom left'], node['custom right'], node['custom top'], node['custom bottom'] = self.unpack_from('4H', data)
		framenum, node['flags'], contentoffset, frametableoffset = self.unpack_from('2B2x2I', data)
		# Window content data
		node['top left vertex'], node['top right vertex'], node['bottom left vertex'], node['bottom right vertex'] = self.unpack_from('(4B)(4B)(4B)(4B)', data, startpos + contentoffset)
		material, coordnum = self.unpack_from('HBx', data)
		node['material'] = self.matnames[material]
		node['uv coordinates'] = []
		for i in range(coordnum):
			coordnode = OrderedDict()
			coordnode['top left'], coordnode['top right'], coordnode['bottom left'], coordnode['bottom right'] = self.unpack_from('(2f)(2f)(2f)(2f)', data)
			node['uv coordinates'].append(coordnode)
		# Window frames
		offsets = self.unpack_from('%dI' % framenum, data, startpos + frametableoffset)
		node['frames'] = []
		for offset in offsets:
			framenode = OrderedDict()
			material, texflip = self.unpack_from('HBx', data, startpos + offset)
			framenode['material'] = self.matnames[material]
			framenode['texture flip'] = texflip
			node['frames'].append(framenode)
		return 'wnd1 - ' + node['name'], node

	def readtxt1(self, data, startpos):
		node = OrderedDict()
		magic, size = self.unpack_from('4sI', data)
		self.readpane(data, node)
		node['length'], node['restricted length'], material, font, align, linealign, flags = self.unpack_from('4H3Bx', data)
		node['material'] = self.matnames[material]
		node['font'] = self.fontnames[font]
		node['x alignment'] = ORIG_X[align & 0b00000011]
		node['y alignment'] = ORIG_Y[(align & 0b00001100) >> 2]
		node['line alignment'] = TEXT_ALIGNS[linealign]
		node['shadow enabled'] = bool(flags & 0b00000001)
		node['restricted length enabled'] = bool(flags & 0b00000010)
		node['invisible border'] = bool(flags & 0b00000100)
		node['two cycles border rendering'] = bool(flags & 0b00001000)
		node['per char transform enabled'] = bool(flags & 0b00010000)
		node['unknown'] = bool(flags & 0b00100000)
		node['italic tilt'], textoffset, node['font top color'], node['font bottom color'], node['x font size'], node['y font size'] = self.unpack_from('fI (4B)(4B)2f', data)
		node['char spacing'], node['line spacing'], callnameoffset, node['shadow x'], node['shadow y'], node['shadow width'], node['shadow height'] = self.unpack_from('2f I 4f', data)
		node['shadow top color'], node['shadow bottom color'], node['shadow italic tilt'], chartransformoffset = self.unpack_from('(4B)(4B)fI', data)
		if callnameoffset != 0:
			node['call name'] = self.unpack_from('n', data, startpos + callnameoffset)[0].decode('utf-8')
		# The null in UTF-16 is a double null... So let's decode by hand...
		data.seek(startpos + textoffset)
		text = b''
		curchar = data.read(2)
		while curchar != b'\x00\x00':
			text += curchar
			curchar = data.read(2)
		node['text'] = text.decode('utf-16-' + ('le' if self.byteorder == '<' else 'be'))
		# TODO : Researches about the per-character transform structure (probably a transform matrix or something)
		if chartransformoffset != 0:
			node['per character transform'] = self.unpack_from('48s', data, startpos + chartransformoffset)[0].hex()
		return 'txt1 - ' + node['name'], node

	def readpic1(self, data, startpos):
		node = OrderedDict()
		magic, size = self.unpack_from('4sI', data)
		self.readpane(data, node)
		node['top left vertex'], node['top right vertex'], node['bottom left vertex'], node['bottom right vertex'] = self.unpack_from('(4B)(4B)(4B)(4B)', data)
		material, coordnum = self.unpack_from('HBx', data)
		node['material'] = self.matnames[material]
		node['uv coordinates'] = []
		for i in range(coordnum):
			coordnode = OrderedDict()
			coordnode['top left'], coordnode['top right'], coordnode['bottom left'], coordnode['bottom right'] = self.unpack_from('(2f)(2f)(2f)(2f)', data)
			node['uv coordinates'].append(coordnode)
		return 'pic1 - ' + node['name'], node

	def readprt1(self, data, startpos):
		node = OrderedDict()
		magic, size = self.unpack_from('4sI', data)
		self.readpane(data, node)
		subnum, node['x part scale'], node['y part scale'] = self.unpack_from('I2f', data)
		node['entries'] = []
		for i in range(subnum):
			entry = OrderedDict()
			name, unk1, flags, unk2, paneoffset, cploffset, extraoffset = self.unpack_from('n24a 2BH 3I', data)
			entry['name'] = name.decode('utf-8')
			entry['flags'] = flags
			entry['unk1'] = unk1
			entry['unk2'] = unk2
			pos = data.tell()
			if paneoffset != 0:
				magic, size = self.unpack_from('4sI', data, startpos + paneoffset)
				data.seek(startpos + paneoffset)
				name, subnode = self.readsection(data, magic, startpos + paneoffset)
				entry['pane name'] = name
				entry['pane'] = subnode
			if cploffset != 0:
				magic, size = self.unpack_from('4sI', data, startpos + cploffset)
				data.seek(startpos + cploffset)
				name, subnode = self.readsection(data, magic, startpos + cploffset)
				entry['complement name'] = name
				entry['complement'] = subnode
			if extraoffset != 0:
				entry['extra'] = self.unpack_from('48s', data, startoffset + extraoffset)[0].hex()
			node['entries'].append(entry)
			data.seek(pos)
		if self.version >= 0x08000000:
			node['part name'] = self.unpack_from('n4a', data)[0].decode('utf-8')
		return 'prt1 - ' + node['name'], node

	def readgrp1(self, data, startpos):
		node = OrderedDict()
		if self.version > 0x05020000:
			magic, size, name, childrennum, children = self.unpack_from('4sI |n34a H/p1(|n24a)', data)
		else:
			magic, size, name, childrennum, children = self.unpack_from('4sI |n24a H2x/p1(|n24a)', data)
		node['name'] = name.decode('utf-8')
		node['children'] = [child.decode('utf-8') for child in children]
		self.curparent['__lastgroup'] = node['name']
		return 'grp1 - ' + node['name'], node

	def readgrs1(self, data, startpos):
		node = OrderedDict()
		return 'grs1 - ' + self.curparent['__lastgroup'], node

	def readgre1(self, data, startpos):
		node = OrderedDict()
		return 'gre1 - ' + self.curparent['__lastgroup'], node

	def readusd1(self, data, startpos):
		node = OrderedDict()
		magic, size, entrynum, node['unknown'] = self.unpack_from('4sI 2H', data)
		node['entries'] = []
		for i in range(entrynum):
			entry = OrderedDict()
			entrypos = data.tell()
			nameoffset, dataoffset, length, datatype, unknown = self.unpack_from('2IH2B', data)
			pos = data.tell()
			entry['name'] = self.unpack_from('n', data, entrypos + nameoffset)[0].decode('utf-8')
			entry['type'] = datatype
			entry['unk1'] = unknown
			if datatype == 0:  # String
				entry['data'] = self.unpack_from('%ds' % length, data, entrypos + dataoffset)[0].decode('utf-8')
			elif datatype == 1:  # int32
				entry['data'] = self.unpack_from('%di' % length, data, entrypos + dataoffset)
			elif datatype == 2:  # float
				entry['data'] = self.unpack_from('%df' % length, data, entrypos + dataoffset)
			elif datatype == 3:  # Strings
				entry['unk2'], stringnum, offsets = self.unpack_from('(2H2I) I/p1(I)', data, entrypos + dataoffset)
				entry['data'] = []
				for offset in offsets:
					entry['data'].append(self.unpack_from('n', data, entrypos + dataoffset + 8 + offset)[0].decode('utf-8'))
			data.seek(pos)
			node['entries'].append(entry)
		if '__lastpane' in self.curparent:
			return 'usd1 - ' + self.curparent['__lastpane'], node
		else:
			return 'usd1 - ' + tuple(self.curparent.keys())[-1], node

	def readcnt1(self, data, startpos):
		node = OrderedDict()
		magic, size, nameoffset, maintableoffset, partnum, animnum, partstableoffset, animtableoffset = self.unpack_from('4sI 2I2H2I', data)
		nameduplicate = self.unpack_from('n20a', data)[0].decode('utf-8')
		node['name'] = self.unpack_from('n', data, startpos + nameoffset)[0].decode('utf-8')
		node['name duplicate'] = nameduplicate

		# Main table
		node['part names'] = []
		data.seek(startpos + maintableoffset)
		for i in range(partnum):
			node['part names'].append(self.unpack_from('n24a', data)[0].decode('utf-8'))
		reference = data.tell()
		offsets = self.unpack_from('%dI' % animnum, data)
		node['animation names'] = []
		for offset in offsets:
			node['animation names'].append(self.unpack_from('n', data, reference + offset)[0].decode('utf-8'))

		# Parts table
		offsets = self.unpack_from('%dI' % partnum, data, startpos + partstableoffset)
		node['parts'] = []
		for offset in offsets:
			node['parts'].append(self.unpack_from('n', data, startpos + partstableoffset + offset)[0].decode('utf-8'))

		# Anims table
		offsets = self.unpack_from('%dI' % animnum, data, startpos + animtableoffset)
		node['animations'] = []
		for offset in offsets:
			node['animations'].append(self.unpack_from('n', data, startpos + animtableoffset + offset)[0].decode('utf-8'))
		return 'cnt1', node
