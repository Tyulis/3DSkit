# -*- coding:utf-8 -*-
import os
from util import error
from util.funcops import byterepr, ClsFunc
from util.fileops import *
import util.rawutil as rawutil
import util.txtree as txtree
from collections import OrderedDict

#Structural, to improve readability
rawutil.register_sub('<', '')
rawutil.register_sub('>', '')

EXTHEADER_MAIN_STRUCT = '512s 512s 256x 256x 512s'
EXTHEADER_SCI_STRUCT = '8s 5s BH (3I) I (3I) I (3I) I (48Q) (2Q48s)'
EXTHEADER_ACI_STRUCT = '(QI 4B (16H) (3Q Q) 34[8s] 15s B) ((28I) 16s) (Q 7s B)'

NEW3DS_SYSTEM_MODES = {
	0: 'Legacy (64MB)',
	1: 'Prod (124MB)',
	2: 'Dev1 (178MB)',
	3: 'Dev2 (124MB)',
}
OLD3DS_SYSTEM_MODES = {
	0: 'Prod (64MB)',
	2: 'Dev1 (96MB)',
	3: 'Dev2 (80MB)',
	4: 'Dev3 (72MB)',
	5: 'Dev4 (32MB)',
}
RESOURCE_LIMIT_CATEGORIES = {
	0: 'Application',
	1: 'System-Applet',
	2: 'Lib-Applet',
	3: 'Other',
}
MEMORY_TYPES = {
	1: 'Application',
	2: 'System',
	3: 'Base',
}

class extractExtHeader (rawutil.TypeReader, ClsFunc):
	def main(self, filename, data, opts={}):
		self.outfile = make_outfile(filename, 'txt')
		self.byteorder = '<'
		self.tree = OrderedDict()
		self.tree['SCI'] = OrderedDict()
		self.tree['ACI'] = OrderedDict()
		self.tree['ACI-limiter'] = OrderedDict()
		self.split(data)
		self.extract_sci()
		self.extract_aci(self.aci, self.tree['ACI'])
		self.extract_aci(self.limit_aci, self.tree['ACI-limiter'])
		self.tree['AccessDesc-Signature'] = self.accessdesc_signature
		self.tree['NCCH-Header-public-key'] = self.public_key
		final = txtree.dump(self.tree)
		write(final, self.outfile)
	
	def split(self, data):
		self.sci, self.aci, self.accessdesc_signature, self.public_key, self.limit_aci = self.unpack(EXTHEADER_MAIN_STRUCT, data)
	
	def extract_sci(self):
		root = self.tree['SCI']
		data = self.unpack(EXTHEADER_SCI_STRUCT, self.sci)
		root['App-Title'] = data[0].rstrip(b'\x00').decode('ascii')
		reserved = data[1]
		flags = data[2]
		root['Compress-ExeFS-code'] = bool(flags & 1)
		root['SD-Application'] = bool(flags & 2)
		root['Remaster-version'] = data[3]
		text_codesetinfo = data[4]
		root['Stack-size'] = data[5]
		ro_codesetinfo = data[6]
		reserved = data[7]
		data_codesetinfo = data[8]
		root['BSS-size'] = data[9]
		dependencies = data[10]
		system_info = data[11]
		
		root['Text-code-set-info'] = OrderedDict()
		sub = root['Text-code-set-info']
		sub['Adress'] = txtree.hexformat(text_codesetinfo[0], 8)
		sub['Physical-region-size'] = text_codesetinfo[1]
		sub['Size'] = text_codesetinfo[2]
		
		root['Read-Only-code-set-info'] = OrderedDict()
		sub = root['Read-Only-code-set-info']
		sub['Adress'] = txtree.hexformat(ro_codesetinfo[0], 8)
		sub['Physical-region-size'] = ro_codesetinfo[1]
		sub['Size'] = ro_codesetinfo[2]
		
		root['Data-code-set-info'] = OrderedDict()
		sub = root['Data-code-set-info']
		sub['Adress'] = txtree.hexformat(data_codesetinfo[0], 8)
		sub['Physical-region-size'] = data_codesetinfo[1]
		sub['Size'] = data_codesetinfo[2]
		
		root['Dependencies'] = [rawutil.hex(el, 16) for el in dependencies if el != 0]
		root['System-Info'] = OrderedDict()
		sys = root['System-Info']
		sys['Savedata-size'] = system_info[0]
		sys['Jump-ID'] = rawutil.hex(system_info[1], 16)
		reserved = system_info[2]
	
	def extract_aci(self, raw, root):
		root['ARM11-Local-system-capabilities'] = OrderedDict()
		arm11_sys = root['ARM11-Local-system-capabilities']
		root['ARM11-Kernel-capabilities'] = OrderedDict()
		arm11_kernel = root['ARM11-Kernel-capabilities']
		root['ARM9-Access-control'] = OrderedDict()
		arm9_access = root['ARM9-Access-control']
		data = self.unpack(EXTHEADER_ACI_STRUCT, raw)
		arm11_sys['ProgramID'] = rawutil.hex(data[0][0], 16)
		arm11_sys['Core-version'] = rawutil.hex(data[0][1], 8)
		flag1, flag2, flag0, priority = data[0][2:6]
		arm11_sys['Enable-L2-cache'] = bool(flag1 & 1)
		arm11_sys['CPU-speed-804MHz'] = bool(flag1 & 2)
		arm11_sys['New3DS-system-mode'] = NEW3DS_SYSTEM_MODES[flag2 & 0b111]
		arm11_sys['Ideal-processor'] = flag0 & 0b11
		arm11_sys['Affinity-mask'] = (flag0 & 0b1100) >> 2
		arm11_sys['Old3DS-system-mode'] = OLD3DS_SYSTEM_MODES[(flag0 & 0b11110000) >> 4]
		arm11_sys['Priority'] = priority
		arm11_sys['Resources-limits-descriptors'] = data[0][6]
		arm11_sys['Storage-info'] = OrderedDict()
		stor = arm11_sys['Storage-info']
		sub = data[0][7]
		stor['ExtData-ID'] = rawutil.hex(sub[0], 16)
		stor['System-Savedata-ID'] = rawutil.hex(sub[1], 16)
		stor['Accessible-unique-ID'] = rawutil.hex(sub[2], 16)
		fs_access = sub[3]
		stor['FileSystem-access-info'] = self.extract_fsaccess(fs_access)
		otherattrs = (fs_access & 0xff00000000000000) >> 56
		stor['Use-RomFS'] = not bool(otherattrs & 1)
		stor['Use-Extended-Savedata-access'] = bool(otherattrs & 2)
		arm11_sys['Services-access'] = [el[0].rstrip(b'\x00').decode('ascii') for el in data[0][8] if el[0] != bytes(8)]
		reserved = data[0][9]
		arm11_sys['Resource-limit-category'] = RESOURCE_LIMIT_CATEGORIES[data[0][10]]
		
		map_adressrange = []
		descs = data[1][0]
		reserved = data[1][1]
		for desc in descs:
			type = (desc & 0xfff00000) >> 20
			if not type & 0b000100000000:
				arm11_kernel['Interrupt-info'] = desc & 0x0fffffff
			elif not type & 0b000010000000:
				arm11_kernel['System-call-mask-table-index'] = (desc & 0x03000000) >> 24
				arm11_kernel['System-call-mask'] = desc & 0x00ffffff
			elif not type & 0b000000100000:
				arm11_kernel['Kernel-Major-Version'] = (desc & 0xff00) >> 8
				arm11_kernel['Kernel-Minor-Version'] = desc & 0xff
			elif not type & 0b000000010000:
				arm11_kernel['Handle-Table-size'] = desc & 0x3ffff
			elif not type & 0b000000001000:
				arm11_kernel['Kernel-Flags'] = self.extract_kernelflags(desc)
			elif not type & 0b000000000110:
				map_adressrange.append(desc & 0x7ffff)
			elif not type & 0b000000000001:
				arm11_kernel['Mapped-Memory-page'] = desc & 0x7ffff
				arm11_kernel['Mapped-Memory-Read-Only'] = bool(desc & 0x100000)
			if map_adressrange != []:
				arm11_kernel['Mapped-Adress-range'] = map_adressrange
		
		desc = data[2][0]
		arm9_access['Accesses'] = self.extract_arm9accesses(desc)
		arm9_access['ARM9-Descriptor-version'] = data[2][2]
	
	def extract_fsaccess(self, flags):
		perms = []
		if flags & 0x000001:
			perms.append('Category-System-App')
		if flags & 0x000002:
			perms.append('Category-Hardware-Check')
		if flags & 0x000004:
			perms.append('Category-FileSystem-Tool')
		if flags & 0x000008:
			perms.append('Debug')
		if flags & 0x000010:
			perms.append('TWL-Card-Backup')
		if flags & 0x000020:
			perms.append('TWL-NAND-Data')
		if flags & 0x000040:
			perms.append('BOSS')
		if flags & 0x000080:
			perms.append('sdmc:/')
		if flags & 0x000100:
			perms.append('Core')
		if flags & 0x000200:
			perms.append('nand:/ro/ (Read-Only)')
		if flags & 0x000400:
			perms.append('nand:/rw/')
		if flags & 0x000800:
			perms.append('nand:/ro/ (Write-Only)')
		if flags & 0x001000:
			perms.append('Category-System-Settings')
		if flags & 0x002000:
			perms.append('CardBoard')
		if flags & 0x004000:
			perms.append('IVS-Import/Export')
		if flags & 0x008000:
			perms.append('sdmc:/ (Write-Only)')
		if flags & 0x010000:
			perms.append('Switch-Cleanup')
		if flags & 0x020000:
			perms.append('Savedata-move')
		if flags & 0x040000:
			perms.append('Shop')
		if flags & 0x080000:
			perms.append('Shell')
		if flags & 0x100000:
			perms.append('Category-Home-Menu')
		if flags & 0x200000:
			perms.append('Seed-DB')
		return perms
	
	def extract_kernelflags(self, desc):
		flags = OrderedDict()
		flags['Allow-debug'] = bool(desc & 0x0001)
		flags['Force-debug'] = bool(desc & 0x0002)
		flags['Allow-non-alphanum'] = bool(desc & 0x0004)
		flags['Shared-page-writing'] = bool(desc & 0x0008)
		flags['Privilege-priority'] = bool(desc & 0x0010)
		flags['Allow-main()-args'] = bool(desc & 0x0020)
		flags['Shared-device-memory'] = bool(desc & 0x0040)
		flags['Runnable-on-sleep'] = bool(desc & 0x0080)
		flags['Memory-type'] = MEMORY_TYPES[(desc & 0x0f00) >> 8]
		flags['Special-memory'] = bool(desc & 0x1000)
		flags['CPU-Core2-access'] = bool(desc & 0x2000)
		return flags
	
	def extract_arm9accesses(self, desc):
		flags = []
		if desc & 0x001:
			flags.append('Mount nand:/')
		if desc & 0x002:
			flags.append('Mount nand:/ro/ (write)')
		if desc & 0x004:
			flags.append('Mount twln:/')
		if desc & 0x008:
			flags.append('Mount wnand:/')
		if desc & 0x010:
			flags.append('Mound card SPI')
		if desc & 0x020:
			flags.append('Use SDIF3')
		if desc & 0x040:
			flags.append('Create seed')
		if desc & 0x080:
			flags.append('SD Application')
		if desc & 0x100:
			flags.append('Mount sdmc:/ (write)')
		return flags
