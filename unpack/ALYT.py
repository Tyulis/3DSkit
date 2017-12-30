# -*- coding:utf-8 -*-
import os
from io import BytesIO
from util import error
from util.utils import byterepr
from util.filesystem import *
import util.rawutil as rawutil
from unpack.SARC import extractSARC
from collections import OrderedDict
from util.txtree import dump

ALYT_META_STRUCT = '4s9I /3s /5s /7s128a I/p1[n64a] I/p1[n32a]128a'
ALYT_LTBL_STRUCT = '4s2H /2[I] /2[4H(3I) /1[H]4a /1[H]4a /2[I]]'


class extractALYT (rawutil.TypeReader):
	def __init__(self, filename, data, verbose, opts={}):
		self.verbose = verbose
		self.outdir = make_outdir(filename)
		try:
			os.mkdir(self.outdir + '_alyt_')
		except:
			pass
		self.metapath = self.outdir + '_alyt_' + os.path.sep
		self.byteorder = '<'
		self.readmeta(data)
		self.extractor = extractSARC(filename, BytesIO(self.sarc), self.verbose)

	def readmeta(self, data):
		meta, ptr = self.unpack_from(ALYT_META_STRUCT, data, 0, getptr=True)
		if meta[0] != b'ALYT':
			error.InvalidMagicError('Invalid magic %s, expected ALYT' % byterepr(meta[0]))
		self.ltbl = meta[10]
		self.lmtl = meta[11]
		self.lfnl = meta[12]
		self.nametable = [el[0].decode('utf-8') for el in meta[14]]
		self.symtable = [el[0].decode('utf-8') for el in meta[16]]
		self.sarc = data.read()

	def extract(self):
		self.extractor.extract()
		self.extractLTBL()
		self.extractLMTL()
		self.extractLFNL()
		self.dumptables()

	def extractLTBL(self):
		bwrite(self.ltbl, self.metapath + 'LTBL.bin')
		ltbl = self.unpack(ALYT_LTBL_STRUCT, self.ltbl)
		entries = ltbl[4]
		final = []
		for edat in entries:
			entry = OrderedDict()
			entry['BFLYT'] = self.nametable[edat[0]]
			entry['unknown'] = edat[3]
			entry['prt1'] = [self.symtable[el[0]] for el in edat[5]]
			entry['links'] = [self.symtable[el[0]] for el in edat[6]]
			entry['animations'] = [self.nametable[el[0]] for el in edat[7]]
			final.append(entry)
		write(dump({'LTBL': final}), self.metapath + 'LTBL.txt')

	def extractLMTL(self):
		bwrite(self.lmtl, self.metapath + 'LMTL.bin')

	def extractLFNL(self):
		bwrite(self.lfnl, self.metapath + 'LFNL.bin')

	def dumptables(self):
		write('\n'.join(self.nametable), self.metapath + 'nametable.txt')
		write('\n'.join(self.symtable), self.metapath + 'symtable.txt')
