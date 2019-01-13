# -*- coding:utf-8 -*-
import os
import json
from collections import OrderedDict
from util import error, ENDIANS
from util.utils import byterepr, ClsFunc
from util.filesystem import *
import util.rawutil as rawutil

TARGETS = {
	b'FLPA': ('X translation', 'Y translation', 'Z translation', 'X rotation', 'Y rotation',
				'Z rotation', 'X scale', 'Y scale', 'X size', 'Y size'),
	b'FLVI': ('Visibility', ),
	b'FLTP': ('Texture pattern', ),
	b'FLVC': ('Top left red', 'Top left green', 'Top left blue', 'Top left alpha',
				'Top right red', 'Top right green', 'Top right blue', 'Top right alpha',
				'Bottom left red', 'Bottom left green', 'Bottom left blue', 'Bottom left alpha',
				'Bottom right red', 'Bottom right green', 'Bottom right blue', 'Bottom right alpha', 'Pane alpha'),
	b'FLMC': ('Black color red', 'Black color green', 'Black color blue', 'Black color alpha',
				'White color red', 'White color green', 'White color blue', 'White color alpha',
				'Texture color blend ratio',
				'Tev color 0 red', 'Tev color 0 green', 'Tev color 0 blue', 'Tev color 0 alpha',
				'Tev color 1 red', 'Tev color 1 green', 'Tev color 1 blue', 'Tev color 1 alpha',
				'Tev color 2 red', 'Tev color 2 green', 'Tev color 2 blue', 'Tev color 2 alpha',
				'Tev konstant color 0 red', 'Tev konstant color 0 green', 'Tev konstant color 0 blue', 'Tev konstant color 0 alpha',
				'Tev konstant color 1 red', 'Tev konstant color 1 green', 'Tev konstant color 1 blue', 'Tev konstant color 1 alpha',
				'Tev konstant color 2 red', 'Tev konstant color 2 green', 'Tev konstant color 2 blue', 'Tev konstant color 2 alpha',),
	b'FLTS': ('U translation', 'V translation', 'Rotation', 'U scale', 'V scale'),
	b'FLIM': ('Rotation', 'U scale', 'V scale'),
	b'FLEU': ('<unknown-FLEU>', ),
}

ANIM_TARGET_TYPES = ('pane', 'material', '<unknown-2>')


class extractBFLAN (rawutil.TypeReader, ClsFunc):
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
		if magic != b'FLAN':
			error.InvalidMagicError('Invalid magic %s, expected FLAN' % byterepr(magic))
		node = self.makenode(self.output, 'FLAN')
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
		for i in range(self.secnum):
			startpos = data.tell()
			magic, size = self.unpack_from('4sI', data)
			data.seek(startpos)
			name, node = self.readsection(data, magic, startpos)
			self.output[name] = node
			data.seek(startpos + size)

	def readsection(self, data, magic, startpos):
		if magic == b'pat1':
			name, node = self.readpat1(data, startpos)
		elif magic == b'pai1':
			name, node = self.readpai1(data, startpos)
		else:
			error.InvalidSectionError('Invalid section magic %s at 0x%08X' % (byterepr(magic), startpos))
		return name, node

	def readpat1(self, data, startpos):
		node = OrderedDict()
		magic, size, node['order'], groupnum, nameoffset, groupnamesoffset, node['file start'], node['file end'], node['child binding'] = self.unpack_from('4sI2H2I2HB3x', data)
		name = self.unpack_from('n', data, startpos + nameoffset)[0]
		node['name'] = name.decode('utf-8')
		data.seek(startpos + groupnamesoffset)
		node['groups'] = []
		for i in range(groupnum):
			name = self.unpack_from('n28a', data)[0]
			node['groups'].append(name.decode('utf-8'))
		return 'pat1', node

	def readpai1(self, data, startpos):
		node = OrderedDict()
		magic, size, node['frame size'], node['flags'], texnum, entrynum, tableoffset = self.unpack_from('4sI HBx 2HI', data)
		basepos = data.tell()
		texnamesoffsets = self.unpack_from('%dI' % texnum, data)
		node['textures'] = []
		for offset in texnamesoffsets:
			name = self.unpack_from('n', data, basepos + offset)[0]
			node['textures'].append(name.decode('utf-8'))
		basepos = data.tell()
		entryoffsets = self.unpack_from('%dI' % entrynum, data, startpos + tableoffset)
		for i, offset in enumerate(entryoffsets):
			animation = self.makenode(node, 'animation %d' % i)
			entrypos = startpos + offset
			name, tagnum, targettype = self.unpack_from('n28a 2B2x', data, entrypos)
			animation['name'] = name.decode('utf-8')
			animation['target type'] = ANIM_TARGET_TYPES[targettype]
			tagoffsets = self.unpack_from('%dI' % tagnum, data)
			for j, tagoffset in enumerate(tagoffsets):
				data.seek(entrypos + tagoffset)
				if targettype == 2:
					unknown = self.unpack_from('I', data)[0]
				tagpos = data.tell()
				magic, tagentrynum = self.unpack_from('4sI', data)
				tag = self.makenode(animation, '%s %d' % (magic.decode('utf-8'), j))
				if targettype == 2:
					tag['unknown'] = unknown
				tagentryoffsets = self.unpack_from('%dI' % tagentrynum, data)
				for k, tagentryoffset in enumerate(tagentryoffsets):
					tagentry = self.makenode(tag, 'entry %d' % k)
					tagentrypos = tagentryoffset + tagpos
					tagentry['index'], target, datatype, framenum, firstframeoffset = self.unpack_from('2B2H2xI', data, tagpos + tagentryoffset)
					tagentry['target'] = TARGETS[magic][target]
					tagentry['data type'] = datatype
					data.seek(tagentrypos + firstframeoffset)
					for l in range(framenum):
						coordnode = self.makenode(tagentry, 'key frame %d' % l)
						if datatype == 2:
							coordnode['frame'], coordnode['value'], coordnode['blend'] = self.unpack_from('3f', data)
						elif datatype == 1:
							coordnode['frame'], coordnode['value'], coordnode['unknown'] = self.unpack_from('f2H', data)
					if magic == b'FLEU':
						namelen, name = self.unpack_from('I/p1s', data)
						tagentry['name'] = name.decode('utf-8')

		return 'pai1', node
