# -*- coding:utf-8 -*-
#Thanks to gbatek...
import os
from util import error, ENDIANS
from util.fileops import *
from util.funcops import split, FreeObject, byterepr
import util.rawutil as rawutil
from util.txtree import dump
from util.image2gif import writeGif
from collections import OrderedDict
from PIL import Image, ImageOps

NDS_HEADER_STRUCT = '12s4sH 3B 8s 3B19I2H2I 8s 2I 56s156s 2H3I 4s10s'
DSI_EXTHEADER_STRUCT = '(5I)(3I)(3I) 26I 12s4I (4s4B)2I 176s (16B) 20s20s20s20s20s20s 40s 20s 2636s384s 128s'
NITRO_FAT_TBL = '{2I}'
NITRO_MAIN_FNT_ENTRY = 'I2H'

UNIT_CODES = {
	0x00: 'NDS',
	0x02: 'NDS + DSI',
	0x03: 'DSI'
}

DEVICE_CAPACITIES = {
	0x00: '128kB',
	0x01: '256kB',
	0x02: '512kB',
	0x03: '1MB',
	0x04: '2MB',
	0x05: '4MB',
	0x06: '8MB',
	0x07: '16MB',
	0x08: '32MB',
	0x09: '64MB',
	0x0a: '128MB',
	0x0b: '256MB',
	0x0c: '512MB'
}

NTR_REGIONS = {
	0x00: 'Normal',
	0x80: 'China',
	0x40: 'Korea'
}

CONTENT_TYPES = {
	0x00: 'Cartridge',
	0x04: 'DSIWare',
	0x05: 'System Fun Tool',
	0x0f: 'Non-executable data archive',
	0x15: 'System Base Tool',
	0x17: 'System Menu'
}

AGE_RATING_TYPES = {
	0x00: 'CERO (Japan)',
	0x01: 'ESRB (USA)',
	0x02: 'Reserved',
	0x03: 'USK (Germany)',
	0x04: 'PEGI (Europe)',
	0x05: 'Reserved',
	0x06: 'PEGI (Portugal)',
	0x07: 'BEFC (UK)',
	0x08: 'AGCB (Australia)',
	0x09: 'GRB (South Korea)',
	0x0a: 'Reserved',
	0x0b: 'Reserved',
	0x0c: 'Reserved',
	0x0d: 'Reserved',
	0x0e: 'Reserved',
	0x0f: 'Reserved',
}

CRC_BASE = 0xffff
CRC_VALUES = (0xc0c1, 0xc181, 0xc301, 0xc601, 0xcc01, 0xd801, 0xf001, 0xa001)


class NitroFATEntry (object):
	def __init__(self):
		self.start = 0
		self.end = 0


class extractNDS (rawutil.TypeReader):
	def __init__(self, filename, data, opts={}):
		self.byteorder = '<'
		self.outdir = make_outdir(filename)
		header = data[:0x4000]
		self.data = data
		self.read_NTRheader(header)
		self.extract_sections(data)
		self.read_FAT()
		self.read_FNT()
	
	def paf(self):
		paf
	
	def crc16(self, data):
		crc = CRC_BASE
		for byte in data:
			crc ^= byte
			for i, val in enumerate(CRC_VALUES):
				carry = crc & 1
				crc >>= 1
				if carry:
					crc ^= val << (7 - i)
			crc &= 0xffff
		return crc
	
	def read_NTRheader(self, data):
		hdata = self.unpack(NDS_HEADER_STRUCT, data)
		self.name = hdata[0].rstrip(b'\x00').decode('utf-8')
		self.gamecode = hdata[1].decode('utf-8')
		self.makercode = hdata[2]
		self.unitcode = UNIT_CODES[hdata[3]]
		self.encryptionseed_select = hdata[4]
		self.devicecapacity = DEVICE_CAPACITIES[hdata[5]]
		self.ntr_region = NTR_REGIONS[hdata[7]]
		self.version = hdata[8]
		self.flags = hdata[9]
		self.autostart = (hdata[9] & 0b00000010) != 0
		self.arm9 = FreeObject()
		self.arm9.offset = hdata[10]
		self.arm9.entryaddress = hdata[11]
		self.arm9.ramaddress = hdata[12]
		self.arm9.size = hdata[13]
		self.arm7 = FreeObject()
		self.arm7.offset = hdata[14]
		self.arm7.entryaddress = hdata[15]
		self.arm7.ramaddress = hdata[16]
		self.arm7.size = hdata[17]
		self.fnt = FreeObject()
		self.fnt.offset = hdata[18]
		self.fnt.size = hdata[19]
		self.fat = FreeObject()
		self.fat.offset = hdata[20]
		self.fat.size = hdata[21]
		self.arm9.overlayoffset = hdata[22]
		self.arm9.overlaysize = hdata[23]
		self.arm7.overlayoffset = hdata[24]
		self.arm7.overlaysize = hdata[25]
		self.normalcontrol_setting = hdata[26]
		self.securecontrol_setting = hdata[27]
		self.iconoffset = hdata[28]
		self.secure = FreeObject()
		self.secure.crc = hdata[29]
		self.secure.transfer_timeout = hdata[30]
		self.arm9.autoload = hdata[31]
		self.arm7.autoload = hdata[32]
		self.secure.disable = hdata[33]
		self.ntr_usedsize = hdata[34]  #only NDS parts
		self.headerlen = hdata[35]  #0x4000
		reserved = hdata[36]
		self.nintendo_logo = hdata[37]
		self.nintendo_logocrc = hdata[38]  #0xCF56
		self.header_crc = hdata[39]  #CRC16 hdr[0x0000 : 0x015D]
		self.debug_romoffset = hdata[40]
		self.debug_size = hdata[41]
		self.debug_ramadress = hdata[42]
		reserved1 = hdata[42]
		reserved2 = hdata[44]
		if self.unitcode in ('DSI', 'NDS + DSI'):
			self.read_TWLextheader(data)
		self.make_checks(data)
	
	def make_checks(self, data):
		if self.headerlen != 0x4000:
			error('Invalid NDS ROM: Invalid header length', 306)
		if self.nintendo_logocrc != 0xcf56:
			error('Invalid NDS ROM: Invalid Nintendo logo CRC', 305)
		if self.crc16(self.nintendo_logo) != self.nintendo_logocrc:
			error('Invalid NDS ROM: Invalid Nintendo logo', 301)
		if self.crc16(data[0:0x15e]) != self.header_crc:
			error('Invalid NDS ROM: Invalid header checksum', 305)
	
	def read_TWLextheader(self, data):
		hdata = self.unpack_from(DSI_EXTHEADER_STRUCT, data, 0x180)
		self.mbk1setting, self.mbk2setting, self.mbk3setting, self.mbk4setting, self.mbk5setting = hdata[0]
		self.arm9.mbk6setting, self.arm9.mbk7setting, self.arm9.mbk8setting = hdata[1]
		self.arm7.mbk6setting, self.arm7.mbk7setting, self.arm7.mbk8setting = hdata[2]
		self.mbk9setting = hdata[3]
		self.regionflags = hdata[4]
		self.accesscontrol = hdata[5]
		self.arm7.scfg_extmask = hdata[6]
		self.dsiflags = hdata[7]
		self.use_bannersav = hdata[7] & 0b00000010
		self.arm9i = FreeObject()
		self.arm9i.offset = hdata[8]
		reserved = hdata[9]
		self.arm9i.ramaddress = hdata[10]
		self.arm9i.size = hdata[11]
		self.arm7i = FreeObject()
		self.arm7i.offset = hdata[12]
		self.params_offset = hdata[13]  #???
		self.arm7i.ramaddress = hdata[14]
		self.arm7i.size = hdata[15]
		self.ntr_hashoffset = hdata[16]
		self.ntr_hashsize = hdata[17]
		self.twl_hashoffset = hdata[18]
		self.twl_hashsize = hdata[19]
		self.sector_hashtable_offset = hdata[20]
		self.sector_hashtable_length = hdata[21]
		self.block_hashtable_offset = hdata[22]
		self.block_hashtable_length = hdata[23]
		self.sectorsize = hdata[24]
		self.block_sectorcount = hdata[25]
		self.icon_bannersize = hdata[26]  #0x23C0
		unknown = hdata[27]
		self.total_usedsize = hdata[28]  #Including TWL parts
		unknown1 = hdata[29]
		self.modcrypt_area1_offset = hdata[30]
		self.modcrypt_area1_length = hdata[31]
		self.modcrypt_area2_offset = hdata[32]
		self.modcrypt_area2_length = hdata[33]
		self.titleid = hdata[34]
		self.publicsav_size = hdata[35]
		self.privatesav_size = hdata[36]
		reserved1 = hdata[37]
		self.age_ratings = hdata[38]
		#SHA1 HMAC hashes
		self.arm9.hash = hdata[39]
		self.arm7.hash = hdata[40]
		self.masterhash = hdata[41]
		self.bannerhash = hdata[42]
		self.arm9i.hash = hdata[43]
		self.arm7i.hash = hdata[44]
		reserved2 = hdata[45]
		self.arm9.hash_without_securearea = hdata[46]
		reserved2 = hdata[47]
		reserved3 = hdata[48]  #used to pass arguments for debug. Usually zeros
		self.rsa_signature = hdata[49]
		
	def extract_sections(self, data):
		self.arm9.data = data[self.arm9.offset: self.arm9.offset + self.arm9.size]
		self.arm7.data = data[self.arm7.offset: self.arm7.offset + self.arm7.size]
		self.fat.data = data[self.fat.offset: self.fat.offset + self.fat.size]
		self.fnt.data = data[self.fnt.offset: self.fnt.offset + self.fnt.size]
		if 'DSI' in self.unitcode:
			self.arm9i.data = data[self.arm9i.offset: self.arm9i.offset + self.arm9i.size]
			self.arm7i.data = data[self.arm7i.offset: self.arm7i.offset + self.arm7i.size]
		self.arm9.overlay = data[self.arm9.overlayoffset: self.arm9.overlayoffset + self.arm9.overlaysize]
		self.arm7.overlay = data[self.arm7.overlayoffset: self.arm7.overlayoffset + self.arm7.overlaysize]
		self.icon = data[self.iconoffset: self.iconoffset + 0x23c0]
	
	def read_FAT(self):
		self.files = []
		fat = self.unpack(NITRO_FAT_TBL, self.fat.data)
		for entry in fat[0]:
			file = NitroFATEntry()
			file.start, file.end = entry
			self.files.append(file)
	
	def read_FNT(self):
		self.tree = {}
		self.read_MainFNTEntry(0xF000, self.tree)
	
	def read_MainFNTEntry(self, id, dir):
		offset = (id & 0x0fff) * 8
		sub_offset, firstfile_id, parent = self.unpack_from(NITRO_MAIN_FNT_ENTRY, self.fnt.data, offset)
		self.read_FNTSubTable(sub_offset, firstfile_id, dir)
	
	def read_FNTSubTable(self, ptr, firstfile, dir):
		actfile = firstfile
		while True:
			type, ptr = self.uint8(self.fnt.data, ptr)
			if type == 0x00:  #end of subtable
				break
			elif type == 0x80:  #???
				continue
			elif type & 0x80:  #subdir
				namelen = type & 0x7f
				name = self.fnt.data[ptr: ptr + namelen].decode('ascii')
				ptr += namelen
				subdir_id, ptr = self.uint16(self.fnt.data, ptr)
				dir[name] = {}
				subdir = dir[name]
				self.read_MainFNTEntry(subdir_id, subdir)
			else:  #file
				namelen = type & 0x7f
				name = self.fnt.data[ptr: ptr + namelen].decode('ascii')
				ptr += namelen
				dir[name] = self.files[actfile]
				actfile += 1
	
	def list(self):
		print(dump(self.tree))
	
	def extract(self):
		romout = self.outdir + 'rom' + os.path.sep
		exeout = self.outdir + 'exe' + os.path.sep
		makedirs(romout)
		makedirs(exeout)
		bwrite(self.arm9.data, exeout + 'arm9.bin')
		bwrite(self.arm7.data, exeout + 'arm7.bin')
		bwrite(self.arm9.overlay, exeout + 'arm9_overlay.bin')
		bwrite(self.arm7.overlay, exeout + 'arm7_overlay.bin')
		bwrite(self.data[0:0x4000], exeout + 'header.bin')
		bwrite(self.data[self.arm9.offset:0x8000], exeout + 'secure.bin')
		bwrite(self.icon, exeout + 'icon.bin')
		if 'DSI' in self.unitcode:
			bwrite(self.arm9i.data, exeout + 'arm9i.bin')
			bwrite(self.arm7i.data, exeout + 'arm7i.bin')
		self.extractIcon(self.icon, exeout)
		self.extractHeader(self.data[0:0x4000], exeout + 'header.txt')
		self.extractDir(self.tree, romout)
	
	def extractDir(self, dir, out):
		for name in dir.keys():
			el = dir[name]
			if isinstance(el, dict):
				subdir = out + name + os.path.sep
				makedirs(subdir)
				self.extractDir(el, subdir)
			else:
				filedata = self.data[el.start: el.end]
				bwrite(filedata, out + name)
	
	def extractIcon(self, data, outdir):
		info = OrderedDict()
		info['animated'] = False
		version, ptr = self.uint16(data, 0)
		crc, ptr = self.uint16(data, ptr)
		if self.crc16(data[0x0020:0x0840]) != crc:
			error('Invalid NDS ROM: Invalid icon CRC', 305)
		if version < 0x0103:
			bitmap, palette = self.unpack_from('512s16[H]', data, 0x20)
			img = self.extractIconBitmap(bitmap, palette)
			img.save(outdir + 'icon.png', 'PNG')
		else:
			frames = []
			pals = []
			bitmap, pal = self.unpack_from('512s16[H]', data, 0x20)
			noanim = self.extractIconBitmap(bitmap, pal)
			ptr = 0x1240
			palptr = 0x2240
			for i in range(0, 8):
				bmp, ptr = self.unpack_from('512s', data, ptr, getptr=True)
				pal, palptr = self.unpack_from('16[H]', data, palptr, getptr=True)
				frames.append(bmp[0])
				pals.append(pal[0])
			tokens = self.unpack_from('64H', data, 0x2340)
			if tokens[0] == 0x0000:
				noanim.save(outdir + 'icon.png', 'PNG')
			else:
				info['animated'] = True
				anim = []
				durations = []
				for tk in tokens:
					vertflip = tk & 0b1000000000000000
					horiflip = tk & 0b0100000000000000
					palindex = tk & 0b0011100000000000
					bmpindex = tk & 0b0000011100000000
					duration = tk & 0b0000000011111111
					palindex >>= 11
					bmpindex >>= 8
					frame = self.extractIconBitmap(frames[bmpindex], pals[palindex])
					if vertflip:
						ImageOps.flip(frame)
					if horiflip:
						ImageOps.mirror(frame)
					anim.append(frame)
					durations.append(duration / 60)
				writeGif(outdir + 'icon.gif', anim, durations)
					
		jp = self.utf16string(data, 0x240)[0]
		en = self.utf16string(data, 0x340)[0]
		fr = self.utf16string(data, 0x440)[0]
		de = self.utf16string(data, 0x540)[0]
		it = self.utf16string(data, 0x640)[0]
		es = self.utf16string(data, 0x740)[0]
		info['title'] = OrderedDict()
		info['title']['JP'] = jp
		info['title']['EN'] = en
		info['title']['FR'] = fr
		info['title']['DE'] = de
		info['title']['IT'] = it
		info['title']['ES'] = es
		if version >= 0x0002:
			cn = self.utf16string(data, 0x840)[0]
			info['title']['CN'] = cn
		if version >= 0x0003:
			kr = self.utf16string(data, 0x940)[0]
			info['title']['KR'] = kr
		write(dump(info), outdir + 'icon.txt')
	
	def extractIconBitmap(self, bitmap, binpal):
		img = Image.new('RGBA', (32, 32))
		pixels = img.load()
		tiles = split(bitmap, 32)
		palette = []
		for el in binpal:
			el = el[0]
			color = [0, 0, 0, 255]
			color[2] = ((el & 0b0111110000000000) >> 10) * 8
			color[1] = ((el & 0b0000001111100000) >> 5) * 8
			color[0] = (el & 0b0000000000011111) * 8
			color = tuple(color)  #stupid tuple forcing...
			palette.append(color)
		palette[0] = (0, 0, 0)
		for tiley in range(0, 4):
			for tilex in range(0, 4):
				tile = tiles[4 * tiley + tilex]
				for y in range(0, 8):
					for x in range(0, 4):
						pxx = (tilex * 8) + (x * 2)
						pxy = (tiley * 8) + y
						pxbs = tile[4 * y + x]
						px2 = palette[(pxbs & 0xf0) >> 4]
						px1 = palette[pxbs & 0x0f]
						pixels[pxx, pxy] = px1
						pixels[pxx + 1, pxy] = px2
		return img
	
	def extractHeader(self, data, outname):
		info = OrderedDict()
		ntr = OrderedDict()
		twl = OrderedDict()
		ntr['name'] = self.name
		ntr['gamecode'] = self.gamecode
		ntr['makercode'] = self.makercode
		ntr['unitcode'] = self.unitcode
		ntr['device-capacity'] = self.devicecapacity
		ntr['region'] = self.ntr_region
		ntr['version'] = self.version
		ntr['encryption-seed-select'] = self.encryptionseed_select
		arm9 = OrderedDict()
		arm7 = OrderedDict()
		arm9['RAM-address'] = self.arm9.ramaddress
		arm9['entry-address'] = self.arm9.entryaddress
		arm9['auto-load'] = self.arm9.autoload
		arm7['RAM-address'] = self.arm7.ramaddress
		arm7['entry-address'] = self.arm7.entryaddress
		arm7['auto-load'] = self.arm7.autoload
		ntr['ARM9'] = arm9
		ntr['ARM7'] = arm7
		ntr['auto-start'] = self.autostart
		ntr['ROM-used-size'] = self.ntr_usedsize
		ntr['normal-control-setting'] = self.normalcontrol_setting
		ntr['secure-control-setting'] = self.securecontrol_setting
		ntr['secure-disable'] = self.secure.disable
		ntr['secure-transfer-timeout'] = self.secure.transfer_timeout
		info['NTR'] = ntr
		if 'DSI' in self.unitcode:
			twl['ARM9i-RAM-address'] = self.arm9i.ramaddress
			twl['ARM7i-RAM-address'] = self.arm9.ramaddress
			mbk = OrderedDict()
			mbk['MBK1'] = self.mbk1setting
			mbk['MBK2'] = self.mbk2setting
			mbk['MBK3'] = self.mbk3setting
			mbk['MBK4'] = self.mbk4setting
			mbk['MBK5'] = self.mbk5setting
			arm9 = OrderedDict()
			arm7 = OrderedDict()
			arm9['MBK6'] = self.arm9.mbk6setting
			arm9['MBK7'] = self.arm9.mbk7setting
			arm9['MBK8'] = self.arm9.mbk8setting
			arm7['MBK6'] = self.arm7.mbk6setting
			arm7['MBK7'] = self.arm7.mbk7setting
			arm7['MBK8'] = self.arm7.mbk8setting
			mbk['MBK9'] = self.mbk9setting
			mbk['ARM9'] = arm9
			mbk['ARM7'] = arm7
			twl['MBK-settings'] = mbk
			twl['ARM7-SCFG-extmask'] = self.arm7.scfg_extmask
			twl['access-control'] = self.accesscontrol
			twl['region'] = self.region(self.regionflags)
			twl['total-used-size'] = self.total_usedsize
			twl['titleID'], twl['content-type'] = self.maketitleid(self.titleid)
			twl['public.sav-size'] = self.publicsav_size
			twl['private.sav-size'] = self.privatesav_size
			twl['age-ratings'] = self.makeageratings(self.age_ratings)
			info['TWL'] = twl
		write(dump(info), outname)
		
	def region(self, flags):
		reg = []
		if flags & 0x01:
			reg.append('JPN')
		if flags & 0x02:
			reg.append('USA')
		if flags & 0x04:
			reg.append('EUR')
		if flags & 0x08:
			reg.append('AUS')
		if flags & 0x10:
			reg.append('CHN')
		if flags & 0x20:
			reg.append('KOR')
		return reg
	
	def maketitleid(self, l):
		l = list(reversed(l))  #stupid non-subscriptable iterators...
		id = rawutil.hex(l[0:4])
		gamecode = list(reversed(list(l[4])))
		id += rawutil.hex(gamecode)
		cntp = CONTENT_TYPES[l[3]]
		return id, cntp
	
	def makeageratings(self, ratings):
		final = {}
		for i, flags in enumerate(ratings):
			type = AGE_RATING_TYPES[i]
			if flags & 0b10000000:
				final[type] = flags & 0x0b00011111
			else:
				final[type] = 'No rating'
			if flags & 0b01000000:
				final[type] = 'Prohibited'
		return final
