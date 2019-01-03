# -*- coding:utf-8 -*-
import json
from collections import OrderedDict
from util import error, BOMS
from util.utils import ClsFunc
import util.rawutil as rawutil
from util.filesystem import *

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
	'<unknown-3', 'pane r and s projection', '<unknown-5>', '<unknown-6>'
)

TEXT_ALIGNS = ('undefined', 'left', 'center', 'right')
ORIG_X = ('center', 'left', 'right')
ORIG_Y = ('center', 'up', 'down')

class packBFLYT(ClsFunc, rawutil.TypeWriter):
	def main(self, filenames, outname, endian, verbose, opts={}):
		self.verbose = verbose
		inname = filenames[0]
		with open(inname, 'r') as f:
			data = json.load(f, object_pairs_hook=OrderedDict)
		self.data = data
		self.byteorder = data['FLYT']['byte order']
		self.version = data['FLYT']['version']
		output = open(outname, 'wb+')
		self.packdata(data, output)
		self.packheader(data, output)
		output.close()

	def packheader(self, data, output):
		output.seek(0, 2)
		filesize = output.tell()
		output.seek(0)
		output.write(b'FLYT' + rawutil.pack('>H', BOMS[self.byteorder]))
		self.pack('H2IH2x', 0x14, data['FLYT']['version'], filesize, self.secnum, output)

	def packdata(self, data, output):
		self.secnum = 0
		output.seek(0x14)  # Skipping the header, it will be written at the end
		for secname in data.keys():
			if secname == 'FLYT':
				continue
			section = data[secname]
			self.packsection(secname, section, output)

	def packsection(self, name, data, output):
		self.secnum += 1
		if name.startswith('lyt1'):
			self.packlyt1(name, data, output)
		elif name.startswith('txl1'):
			self.packtxl1(name, data, output)
		elif name.startswith('fnl1'):
			self.packfnl1(name, data, output)
		elif name.startswith('mat1'):
			self.packmat1(name, data, output)
		elif name.startswith('pan1'):
			self.packpan1(name, data, output)
		elif name.startswith('pas1'):
			self.packpas1(name, data, output)
		elif name.startswith('pae1'):
			self.packpae1(name, data, output)
		elif name.startswith('bnd1'):
			self.packbnd1(name, data, output)
		elif name.startswith('wnd1'):
			self.packwnd1(name, data, output)
		elif name.startswith('txt1'):
			self.packtxt1(name, data, output)
		elif name.startswith('pic1'):
			self.packpic1(name, data, output)
		elif name.startswith('prt1'):
			self.packprt1(name, data, output)
		elif name.startswith('grp1'):
			self.packgrp1(name, data, output)
		elif name.startswith('grs1'):
			self.packgrs1(name, data, output)
		elif name.startswith('gre1'):
			self.packgre1(name, data, output)
		elif name.startswith('usd1'):
			self.packusd1(name, data, output)
		elif name.startswith('cnt1'):
			self.packcnt1(name, data, output)

	def packlyt1(self, name, data, output):
		startpos = output.tell()
		self.pack('8x B3x 4f n', data['drawn from middle'], data['screen width'], data['screen height'], data['max parts width'], data['max parts height'], data['name'], output)

		output.write(b'\x00' * (4 - (output.tell() % 4 or 4)))  # Padding to a multiple of 4
		endpos = output.tell()
		output.seek(startpos)
		self.pack('4sI', b'lyt1', endpos - startpos, output)  # Writing the section header
		output.seek(endpos)

	def packtxl1(self, name, data, output):
		startpos = output.tell()
		self.pack('8x I', len(data['textures']), output)
		baseoffset = 4 * len(data['textures'])
		strings = b''
		offsets = []
		for string in data['textures']:
			offsets.append(baseoffset + len(strings))
			strings += string.encode('utf-8') + b'\x00'
		self.pack('%d(I)' % len(offsets), offsets, output)
		output.write(strings)

		output.write(b'\x00' * (4 - (output.tell() % 4 or 4)))  # Padding to a multiple of 4
		endpos = output.tell()
		output.seek(startpos)
		self.pack('4sI', b'txl1', endpos - startpos, output)  # Writing the section header
		output.seek(endpos)

	def packfnl1(self, name, data, output):
		startpos = output.tell()
		self.pack('8x I', len(data['fonts']), output)
		baseoffset = 4 * len(data['fonts'])
		strings = b''
		offsets = []
		for string in data['fonts']:
			offsets.append(baseoffset + len(strings))
			strings += string.encode('utf-8') + b'\x00'
		self.pack('%d(I)' % len(offsets), offsets, output)
		output.write(strings)

		output.write(b'\x00' * (4 - (output.tell() % 4 or 4)))  # Padding to a multiple of 4
		endpos = output.tell()
		output.seek(startpos)
		self.pack('4sI', b'fnl1', endpos - startpos, output)  # Writing the section header
		output.seek(endpos)

	def packmat1(self, name, data, output):
		startpos = output.tell()
		self.pack('8x I', len(data['materials']), output)
		offsets = []
		offsetspos = output.tell()
		output.seek(len(data['materials']) * 4, 1)  # Skips offset table, written later
		self.matnames = []
		for material in data['materials']:
			offsets.append(output.tell() - startpos)
			self.matnames.append(material['name'])
			self.pack('n28a', material['name'].encode('utf-8'), output)
			texrefs = {}
			textransforms = {}
			mapsettings = {}
			texcombiners = {}
			blendmodes = {}
			blendalpha = {}
			indirect = {}
			alphacompare = {}
			projmappings = {}
			shadowblending = {}
			for setting in material:
				if setting.startswith('texture reference'): texrefs[setting] = material[setting]
				elif setting.startswith('texture transformation'): textransforms[setting] = material[setting]
				elif setting.startswith('mapping setting'): mapsettings[setting] = material[setting]
				elif setting.startswith('texture combiner'): texcombiners[setting] = material[setting]
				elif setting.startswith('alpha comparison'): alphacompare[setting] = material[setting]
				elif setting.startswith('blending mode'): blendmodes[setting] = material[setting]
				elif setting.startswith('alpha blending mode'): blendalpha[setting] = material[setting]
				elif setting.startswith('indirect adjustment'): indirect[setting] = material[setting]
				elif setting.startswith('projection mapping'): projmappings[setting] = material[setting]
				elif setting.startswith('shadow blending'): shadowblending[setting] = material[setting]
			flags = (len(texrefs) + (len(textransforms) << 2) + (len(mapsettings) << 4) + (len(texcombiners) << 6) +
				(len(alphacompare) << 9) + (len(blendmodes) << 10) + (len(blendalpha) << 12) + (len(indirect) << 14) +
				(len(projmappings) << 15) + (len(shadowblending) << 17))
			if material['bad 0x800']:
				flags|= 0x800
			if self.version >= 0x08000000:
				self.pack('I(4B)(4B)(4B)', flags, material['unknown'], material['foreground color'], material['background color'], output)
			else:
				self.pack('(4B)(4B)I', material['foreground color'], material['background color'], flags, output)
			for nodename in sorted(texrefs.keys()):
				node = material[nodename]
				try: texture = self.data['txl1']['textures'].index(node['texture'])
				except: error.BadDataError('In mat1, %s, %s : Texture name %s does not match with any in txl1' % (material['name'], nodename, node['texture']))
				try: wraps = WRAPS.index(node['wrap s'])
				except: error.BadDataError('In mat1, %s, %s : Bad wrapping type %s' % (material['name'], nodename, node['wrap s']))
				try: wrapt = WRAPS.index(node['wrap t'])
				except: error.BadDataError('In mat1, %s, %s : Bad wrapping type %s' % (material['name'], nodename, node['wrap t']))
				self.pack('H2B', texture, wraps, wrapt, output)
			for nodename in sorted(textransforms.keys()):
				node = material[nodename]
				self.pack('5f', node['x translation'], node['y translation'], node['rotation'], node['x scale'], node['y scale'], output)
			for nodename in sorted(mapsettings.keys()):
				node = material[nodename]
				try: method = MAPPING_METHODS.index(node['method'])
				except: error.BadDataError('In mat1, %s, %s : Bad mapping method' % (material['name'], nodename, node['method']))
				if self.version < 0x08000000:
					self.pack('2B(6B)', node['unk1'], method, node['unk2'], output)
				else:
					self.pack('2B(14B)', node['unk1'], method, node['unk2'], output)
			for nodename in sorted(texcombiners.keys()):
				node = material[nodename]
				try: colorblend = COLOR_BLENDS.index(node['color blending'])
				except: error.BadDataError('In mat1, %s, %s : Bad color blending method' % (material['name'], nodename, node['color blending']))
				try: alphablend = ALPHA_BLENDS.index(node['alpha blending'])
				except: error.BadDataError('In mat1, %s, %s : Bad alpha blending method' % (material['name'], nodename, node['alpha blending']))
				self.pack('4B', colorblend, alphablend, node['unk1'], node['unk2'], output)
			for nodename in sorted(alphacompare.keys()):
				node = material[nodename]
				try: condition = ALPHA_COMPARE_CONDITIONS.index(node['condition'])
				except: error.BadDataError('In mat1, %s, %s : Bad alpha comparison condition' % (material['name'], nodename, node['condition']))
				self.pack('4Bf', condition, node['unk1'], node['unk2'], node['unk3'], node['value'], output)
			for nodename in sorted(blendmodes.keys()):
				node = material[nodename]
				try: operation = BLEND_OPS.index(node['operation'])
				except: error.BadDataError('In mat1, %s, %s : Bad blending operation' % (material['name'], nodename, node['operation']))
				try: sourceblend = BLEND_CALC.index(node['source blending'])
				except: error.BadDataError('In mat1, %s, %s : Bad source blending operation' % (material['name'], nodename, node['source blending']))
				try: destblend = BLEND_CALC.index(node['destination blending'])
				except: error.BadDataError('In mat1, %s, %s : Bad destination blending operation' % (material['name'], nodename, node['destination blending']))
				try: logical = LOGICAL_OPS.index(node['logical operation'])
				except: error.BadDataError('In mat1, %s, %s : Bad logical operation' % (material['name'], nodename, node['logical operation']))
				self.pack('4B', operation, sourceblend, destblend, logical, output)
			for nodename in sorted(blendalpha.keys()):
				node = material[nodename]
				try: operation = BLEND_OPS.index(node['operation'])
				except: error.BadDataError('In mat1, %s, %s : Bad blending operation' % (material['name'], nodename, node['operation']))
				try: sourceblend = BLEND_CALC.index(node['source blending'])
				except: error.BadDataError('In mat1, %s, %s : Bad source blending operation' % (material['name'], nodename, node['source blending']))
				try: destblend = BLEND_CALC.index(node['destination blending'])
				except: error.BadDataError('In mat1, %s, %s : Bad destination blending operation' % (material['name'], nodename, node['destination blending']))
				try: logical = LOGICAL_OPS.index(node['logical operation'])
				except: error.BadDataError('In mat1, %s, %s : Bad logical operation' % (material['name'], nodename, node['logical operation']))
				self.pack('4B', operation, sourceblend, destblend, logical, output)
			for nodename in sorted(indirect.keys()):
				node = material[nodename]
				self.pack('3f', node['rotation'], node['x warp'], node['y warp'], output)
			for nodename in sorted(projmappings.keys()):
				node = material[nodename]
				try: option = PROJECTION_MAPPING_TYPES.index(node['option'])
				except: error.BadDataError('In mat1, %s, %s : Bad projection mapping option' % (material['name'], nodename, node['option']))
				self.pack('4f4B', node['x translation'], node['y translation'], node['x scale'], node['y scale'], option, node['unk1'], node['unk2'], node['unk3'], output)
			for nodename in sorted(shadowblending.keys()):
				node = material[nodename]
				self.pack('(4B)(3B)x', node['black blending'], node['white blending'], output)

		endpos = output.tell()
		output.seek(offsetspos)
		self.pack('%d(I)' % len(data['materials']), offsets, output)
		output.seek(endpos)
		output.write(b'\x00' * (4 - (output.tell() % 4 or 4)))  # Padding to a multiple of 4
		endpos = output.tell()
		output.seek(startpos)
		self.pack('4sI', b'mat1', endpos - startpos, output)  # Writing the section header
		output.seek(endpos)

	def packpane(self, name, data, output):
		flags = data['visible'] | (data['transmit alpha'] << 1) | (data['position adjustment'] << 2)
		try: xorigin = ORIG_X.index(data['x origin'])
		except: error.BadDataError('In %s : Bad X origin name %s' % (name, data['x origin']))
		try: yorigin = ORIG_Y.index(data['y origin'])
		except: error.BadDataError('In %s : Bad Y origin name %s' % (name, data['y origin']))
		try: parentxorigin = ORIG_X.index(data['parent x origin'])
		except: error.BadDataError('In %s : Bad parent X origin name %s' % (name, data['parent x origin']))
		try: parentyorigin = ORIG_Y.index(data['parent y origin'])
		except: error.BadDataError('In %s : Bad parent Y origin name %s' % (name, data['parent y origin']))
		origin = xorigin | (yorigin << 2) | (parentxorigin << 4) | (parentyorigin << 6)
		self.pack('4B |n32a 10f', flags, origin, data['alpha'], data['scale'], data['name'].encode('utf-8'), data['x translation'], data['y translation'], data['z translation'], data['x rotation'], data['y rotation'], data['z rotation'], data['x scale'], data['y scale'], data['width'], data['height'], output)

	def packpan1(self, name, data, output):
		startpos = output.tell()
		self.pack('8x', output)
		self.packpane(name, data, output)

		output.write(b'\x00' * (4 - (output.tell() % 4 or 4)))  # Padding to a multiple of 4
		endpos = output.tell()
		output.seek(startpos)
		self.pack('4sI', b'pan1', endpos - startpos, output)  # Writing the section header
		output.seek(endpos)

	def packpas1(self, name, data, output):
		self.pack('4sI', b'pas1', 8, output)  # Writing the section header
		for subsection in data:
			self.packsection(subsection, data[subsection], output)

	def packpae1(self, name, data, output):
		self.pack('4sI', b'pae1', 8, output)  # Writing the section header

	def packbnd1(self, name, data, output):
		startpos = output.tell()
		self.pack('8x', output)
		self.packpane(name, data, output)

		output.write(b'\x00' * (4 - (output.tell() % 4 or 4)))  # Padding to a multiple of 4
		endpos = output.tell()
		output.seek(startpos)
		self.pack('4sI', b'bnd1', endpos - startpos, output)  # Writing the section header
		output.seek(endpos)

	def packwnd1(self, name, data, output):
		startpos = output.tell()
		self.pack('8x', output)
		self.packpane(name, data, output)

		self.pack('8H', data['left stretch'], data['right stretch'], data['top stretch'], data['bottom stretch'], data['custom left'], data['custom right'], data['custom top'], data['custom bottom'], output)
		self.pack('2B2x', len(data['frames']), data['flags'], output)
		offsetspos = output.tell()
		self.pack('8x', output)

		# Window content
		contentoffset = output.tell() - startpos
		self.pack('(4B)(4B)(4B)(4B)', data['top left vertex'], data['top right vertex'], data['bottom left vertex'], data['bottom right vertex'], output)
		try: material = self.matnames.index(data['material'])
		except: error.BadDataError('In %s : Material name %s does not match with any in mat1' % (name, data['material']))
		self.pack('HBx', material, len(data['uv coordinates']), output)
		for coords in data['uv coordinates']:
			self.pack('(2f)(2f)(2f)(2f)', node['top left'], node['top right'], node['bottom left'], node['bottom right'], output)

		# Window frames
		framesoffset = output.tell() - startpos
		offsets = [framesoffset + 4 * i for i in range(len(data['frames']))]
		self.pack('%d(I)' % len(data['frames']), offsets, output)
		for frame in data['frames']:
			try: material = self.matnames.index(frame['material'])
			except: error.BadDataError('In %s : Material name %s does not match with any in mat1' % (name, frame['material']))
			self.pack('HBx', material, frame['texture flip'], output)

		endpos = output.tell()
		output.seek(offsetspos)
		self.pack('2I', contentoffset, framesoffset, output)
		output.seek(endpos)

		output.write(b'\x00' * (4 - (output.tell() % 4 or 4)))  # Padding to a multiple of 4
		endpos = output.tell()
		output.seek(startpos)
		self.pack('4sI', b'wnd1', endpos - startpos, output)  # Writing the section header
		output.seek(endpos)

	def packtxt1(self, name, data, output):
		startpos = output.tell()
		self.pack('8x', output)
		self.packpane(name, data, output)

		try: material = self.matnames.index(data['material'])
		except: error.BadDataError('In %s : Material name %s does not match with any in mat1' % (name, data['material']))
		try: font = self.data['fnl1']['fonts'].index(data['font'])
		except: error.BadDataError('In %s : Font name %s does not match with any in fnl1' % (name, data['font']))
		self.pack('4H', data['length'], data['restricted length'], material, font, output)

		try: xalign = ORIG_X.index(data['x alignment'])
		except: error.BadDataError('In %s : Bad X alignment %s' % (name, data['x alignment']))
		try: yalign = ORIG_Y.index(data['y alignment'])
		except: error.BadDataError('In %s : Bad Y alignment %s' % (name, data['y alignment']))
		try: linealign = TEXT_ALIGNS.index(data['line alignment'])
		except: error.BadDataError('In %s : Bad line alignment %s' % (name, data['line alignment']))
		alignment = xalign | (yalign << 2)
		flags = data['shadow enabled'] | (data['restricted length enabled'] << 1) | (data['invisible border'] << 2) | (data['two cycles border rendering'] << 3) | (data['per char transform enabled'] << 4) | (data['unknown'] << 5)
		self.pack('3Bx f', alignment, linealign, flags, data['italic tilt'], output)
		textoffsetpos = output.tell()
		self.pack('4x (4B)(4B) 4f', data['font top color'], data['font bottom color'], data['x font size'], data['y font size'], data['char spacing'], data['line spacing'], output)
		callnameoffsetpos = output.tell()
		self.pack('4x 4f (4B)(4B)f', data['shadow x'], data['shadow y'], data['shadow width'], data['shadow height'], data['shadow top color'], data['shadow bottom color'], data['shadow italic tilt'], output)
		chartransformoffsetpos = output.tell()
		self.pack('4x', output)
		textoffset = output.tell() - startpos
		textdata = data['text'].encode('utf-16-' + ('le' if self.byteorder == '<' else 'be')) + b'\x00\x00'
		self.pack('%ds4a' % len(textdata), textdata, output)
		if 'call name' in data:
			callnameoffset = output.tell() - startpos
			self.pack('n4a', data['call name'].encode('utf-8'), output)
		else:
			callnameoffset = 0
		if 'per character transform' in data:
			chartransformoffset = output.tell() - startpos
			output.write(bytes.fromhex(data['per character transform']))
		else:
			chartransformoffset = 0

		endpos = output.tell()
		output.seek(textoffsetpos)
		self.pack('I', textoffset, output)
		output.seek(callnameoffsetpos)
		self.pack('I', callnameoffset, output)
		output.seek(chartransformoffsetpos)
		self.pack('I', chartransformoffset, output)
		output.seek(endpos)

		output.write(b'\x00' * (4 - (output.tell() % 4 or 4)))  # Padding to a multiple of 4
		endpos = output.tell()
		output.seek(startpos)
		self.pack('4sI', b'txt1', endpos - startpos, output)  # Writing the section header
		output.seek(endpos)

	def packpic1(self, name, data, output):
		startpos = output.tell()
		self.pack('8x', output)
		self.packpane(name, data, output)

		self.pack('(4B)(4B)(4B)(4B)', data['top left vertex'], data['top right vertex'], data['bottom left vertex'], data['bottom right vertex'], output)
		try: material = self.matnames.index(data['material'])
		except: error.BadDataError('In %s : Material name %s does not match with any in mat1' % (name, data['material']))
		self.pack('HBx', material, len(data['uv coordinates']), output)
		for coords in data['uv coordinates']:
			self.pack('(2f)(2f)(2f)(2f)', coords['top left'], coords['top right'], coords['bottom left'], coords['bottom right'], output)

		output.write(b'\x00' * (4 - (output.tell() % 4 or 4)))  # Padding to a multiple of 4
		endpos = output.tell()
		output.seek(startpos)
		self.pack('4sI', b'pic1', endpos - startpos, output)  # Writing the section header
		output.seek(endpos)

	def packprt1(self, name, data, output):
		startpos = output.tell()
		self.pack('8x', output)
		self.packpane(name, data, output)

		self.pack('I2f', len(data['entries']), data['x part scale'], data['y part scale'], output)
		paneoffsets = {}
		cploffsets = {}
		extraoffsets = {}
		tablepos = output.tell()
		output.seek(40 * len(data['entries']), 1)
		if self.version >= 0x08000000:
			self.pack('n4a', data['part name'], output)
		for i, entry in enumerate(data['entries']):
			if 'pane' in entry:
				paneoffsets[i] = output.tell() - startpos
				self.packsection(entry['pane name'], entry['pane'], output)
				self.secnum -= 1  # prt1 subsections are not counted
			else:
				paneoffsets[i] = 0
		for i, entry in enumerate(data['entries']):
			if 'complement' in entry:
				cploffsets[i] = output.tell() - startpos
				self.packsection(entry['complement name'], entry['complement'], output)
				self.secnum -= 1  # prt1 subsections are not counted
			else:
				cploffsets[i] = 0
		for i, entry in enumerate(data['entries']):
			if 'extra' in entry:
				extraoffsets[i] = output.tell() - startpos
				output.write(bytes.fromhex(entry['extra']))
			else:
				extraoffsets[i] = 0
		endpos = output.tell()

		output.seek(tablepos)
		for i, entry in enumerate(data['entries']):
			self.pack('n24a 2BH3I', entry['name'].encode('utf-8'), entry['unk1'], entry['flags'], entry['unk2'], paneoffsets[i], cploffsets[i], extraoffsets[i], output)
		output.seek(endpos)

		output.write(b'\x00' * (4 - (output.tell() % 4 or 4)))  # Padding to a multiple of 4
		endpos = output.tell()
		output.seek(startpos)
		self.pack('4sI', b'prt1', endpos - startpos, output)  # Writing the section header
		output.seek(endpos)

	def packgrp1(self, name, data, output):
		startpos = output.tell()
		self.pack('8x', output)

		if self.version <= 0x05020000:
			self.pack('n24a H2x', data['name'], len(data['children']), output)
		else:
			self.pack('n34a H', data['name'], len(data['children']), output)
		for child in data['children']:
			self.pack('n24a', child, output)

		output.write(b'\x00' * (4 - (output.tell() % 4 or 4)))  # Padding to a multiple of 4
		endpos = output.tell()
		output.seek(startpos)
		self.pack('4sI', b'grp1', endpos - startpos, output)  # Writing the section header
		output.seek(endpos)

	def packgrs1(self, name, data, output):
		self.pack('4sI', b'grs1', 8, output)
		for subsection in data:
			self.packsection(subsection, data[subsection], output)

	def packgre1(self, name, data, output):
		self.pack('4sI', b'gre1', 8, output)

	def packusd1(self, name, data, output):
		startpos = output.tell()
		self.pack('8x 2H', len(data['entries']), data['unknown'], output)
		names = [entry['name'] for entry in data['entries']]
		nametbl = b''
		datatbl = b''
		nameoffsets = []
		dataoffsets = []
		for entry in data['entries']:
			nameoffsets.append(len(nametbl))
			nametbl += entry['name'].encode('utf-8') + b'\x00'
			dataoffsets.append(len(datatbl))
			if entry['type'] == 0:
				datatbl += entry['data'].encode('utf-8') + b'\x00'
				datatbl += b'\x00' * (4 - (len(datatbl) % 4 or 4))
			elif entry['type'] == 1:
				datatbl += self.pack('%d(i)' % len(entry['data']), entry['data'])
			elif entry['type'] == 2:
				datatbl += self.pack('%d(f)' % len(entry['data']), entry['data'])
			elif entry['type'] == 3:
				datatbl += self.pack('(2H2I)I', entry['unk2'], len(entry['data']))
				stringtbl = b''
				offsets = []
				for string in entry['data']:
					offsets.append(len(stringtbl))
					stringtbl += string.encode('utf-8') + b'\x00'
				stringtbl += b'\x00' * (128 - (len(stringtbl) % 128 or 128))  # FIXME : Why ?
				datatbl += self.pack('%d(I)' % len(entry['data']), [offset + 4 * len(entry['data']) + 8 for offset in offsets])
				datatbl += stringtbl
				datatbl += b'\x00' * (4 - (len(datatbl) % 4 or 4))
		nametbl += b'\x00' * (4 - (len(nametbl) % 4 or 4))

		entrybase = output.tell()
		output.seek(12 * len(data['entries']), 1)
		database = output.tell()
		output.write(datatbl)
		namebase = output.tell()
		output.write(nametbl)
		endpos = output.tell()
		output.seek(entrybase)

		for i, entry in enumerate(data['entries']):
			entrypos = output.tell()
			if entry['type'] == 3:
				length = 1
			else:
				length = len(entry['data'])
			self.pack('2IH2B', nameoffsets[i] + namebase - entrypos, dataoffsets[i] + database - entrypos, length, entry['type'], entry['unk1'], output)

		output.seek(endpos)
		output.write(b'\x00' * (4 - (output.tell() % 4 or 4)))  # Padding to a multiple of 4
		endpos = output.tell()
		output.seek(startpos)
		self.pack('4sI', b'usd1', endpos - startpos, output)  # Writing the section header
		output.seek(endpos)

	def packcnt1(self, name, data, output):
		startpos = output.tell()

		self.pack('28x', output)
		self.pack('n4a', data['name duplicate'], output)
		nameoffset = output.tell() - startpos
		self.pack('n4a', data['name'], output)
		maintableoffset = output.tell() - startpos
		for name in data['part names']:
			self.pack('n24a', name, output)
		reference = output.tell()
		offsets = []
		output.seek(4 * len(data['animation names']), 1)
		for name in data['animation names']:
			offsets.append(output.tell() - reference)
			self.pack('n', name, output)
		endpos = output.tell()
		output.seek(reference)
		self.pack('%d(I)' % len(offsets), offsets, output)
		output.seek(endpos)
		output.write(b'\x00' * (4 - (output.tell() % 4 or 4)))  # Padding to a multiple of 4

		partstablepos = output.tell()
		partstableoffset = partstablepos - startpos
		offsets = []
		output.seek(4 * len(data['parts']), 1)
		for name in data['parts']:
			offsets.append(output.tell() - partstablepos)
			self.pack('n', name, output)
		endpos = output.tell()
		output.seek(partstablepos)
		self.pack('%d(I)' % len(offsets), offsets, output)
		output.seek(endpos)
		output.write(b'\x00' * (4 - (output.tell() % 4 or 4)))  # Padding to a multiple of 4

		animtablepos = output.tell()
		animtableoffset = animtablepos - startpos
		offsets = []
		output.seek(4 * len(data['animations']), 1)
		for name in data['animations']:
			offsets.append(output.tell() - animtablepos)
			self.pack('n', name, output)
		endpos = output.tell()
		output.seek(animtablepos)
		self.pack('%d(I)' % len(offsets), offsets, output)
		output.seek(endpos)
		output.write(b'\x00' * (4 - (output.tell() % 4 or 4)))  # Padding to a multiple of 4

		output.write(b'\x00' * (4 - (output.tell() % 4 or 4)))  # Padding to a multiple of 4
		endpos = output.tell()
		output.seek(startpos)
		self.pack('4sI 2I2H2I', b'cnt1', endpos - startpos, nameoffset, maintableoffset, len(data['parts']), len(data['animations']), partstableoffset, animtableoffset, output)  # Writing the section header
		output.seek(endpos)
