# -*- coding:utf-8 -*-
from util import error
import util.rawutil as rawutil
from util.fileops import *
from util.funcops import ClsFunc, byterepr
from util.wavy import WAV

rawutil.register_sub('S', '(2H2I)')  #sized refs
rawutil.register_sub('R', '(2HI)')  #references
rawutil.register_sub('T', 'I/p1[2HI]')  #ref table

BCSTM_HEADER_STRUCT = '4s2H 2I2H SSS'
BCSTM_INFO_STRUCT = '4sI RRR (4B11IR) $'
BCSTM_TRACK_INFO_STRUCT = '2BHR I/p1[B]'
BCSTM_DSPADPCM_INFO_STRUCT = '8[2h] (2B2h) (2B2h) H'
BCSTM_IMAADPCM_INFO_STRUCT = '(H2B) (H2B)'

PCM8 = 0
PCM16 = 1
DSPADPCM = 2
IMAADPCM = 3

CODECS = {
	0: 'PCM8',
	1: 'PCM16',
	2: 'DSP-ADPCM',
	3: 'IMA-ADPCM',
}


class SizedRef (object):
	def __init__(self, data):
		self.id = data[0]
		self.offset = data[2] if data[2] != 0xffffffff else None
		self.size = data[3]
	
	def getdata(self, data):
		return data[self.offset: self.offset + self.size]


class Reference (object):
	def __init__(self, data):
		self.id = data[0]
		self.offset = data[2] if data[2] != 0xffffffff else None
	
	def __add__(self, obj):
		if self.offset is not None:
			self.offset += obj
		return self


class TrackInfo (object):
	def __init__(self, data):
		self.volume = data[0]
		self.pan = data[1]
		self.channelindex = data[5]


class DSPADPCMContext (object):
	def __init__(self, data):
		self.predictor = data[0] >> 4
		self.scale = data[0] & 0x0f
		self.previous_sample = data[2]
		self.second_previous_sample = data[3]


class DSPADPCMInfo (object):
	def __init__(self, data):
		self.param = data[0]
		self.context = DSPADPCMContext(data[1])
		self.loopcontext = DSPADPCMContext(data[2])


class IMAADPCMContext (object):
	def __init__(self, data):
		self.data = data[0]
		self.table_index = data[1]


class IMAADPCMInfo (object):
	def __init__(self, data):
		self.context = IMAADPCMContext(data[0])
		self.loopcontext = IMAADPCMContext(data[1])


class extractBCSTM (ClsFunc, rawutil.TypeReader):
	def main(self, filename, data, verbose, opts={}):
		self.verbose = verbose
		self.outname = os.path.splitext(filename)[0]
		self.read_header(data)
		self.readINFO()
		self.readSEEK()
		self.readDATA()
	
	def read_header(self, data):
		bom = rawutil.unpack_from('>H', data, 4)[0]
		self.byteorder = '<' if bom == 0xfffe else '>'
		hdata = self.unpack_from(BCSTM_HEADER_STRUCT, data)
		magic = hdata[0]
		if magic != b'CSTM':
			error('Invalid magic %s, expected CSTM' % byterepr(magic))
		bom = hdata[1]
		headerlen = hdata[2]
		self.version = hdata[3]
		filesize = hdata[4]
		blockcount = hdata[5]  #Should be 3
		padding = hdata[6]
		inforef = SizedRef(hdata[7])
		seekref = SizedRef(hdata[8])
		dataref = SizedRef(hdata[9])
		self.info = inforef.getdata(data)
		self.seek = seekref.getdata(data)
		self.data = dataref.getdata(data)
		if self.verbose:
			print('Version: %08x' % self.version)
	
	def readINFO(self):
		data = self.unpack(BCSTM_INFO_STRUCT, self.info)
		info = self.info
		streaminforef = Reference(data[2])
		trackinforef = Reference(data[3]) + 8
		channelinforef = Reference(data[4]) + 8
		streaminfo = data[5]
		self.read_streaminfo(streaminfo)
		trackinforeftable = []
		channelinforeftable = []
		if trackinforef.offset is not None:
			count, ptr = self.uint32(info, trackinforef.offset)
			for i in range(count):
				ref, ptr = self.unpack_from('R', info, ptr, getptr=True)
				ref = Reference(ref[0])
				if ref.offset is None:
					break
				else:
					trackinforeftable.append(ref + trackinforef.offset)
		if channelinforef.offset is not None:
			count, ptr = self.uint32(info, channelinforef.offset)
			for i in range(count):
				ref, ptr = self.unpack_from('R', info, ptr, getptr=True)
				ref = Reference(ref[0])
				if ref.offset is None:
					break
				else:
					channelinforeftable.append(ref + channelinforef.offset)
		self.trackinfo = []
		for ref in trackinforeftable:
			offset = ref.offset
			if offset is not None:
				self.trackinfo.append(TrackInfo(self.unpack_from(BCSTM_TRACK_INFO_STRUCT, self.info, offset)))
		self.channelinfo = []
		for ref in channelinforeftable:
			offset = ref.offset
			if offset is not None:
				adpcminforef = Reference(self.unpack_from('R', self.info, offset)[0])
				if self.codec == DSPADPCM:
					rawentry = self.unpack_from(BCSTM_DSPADPCM_INFO_STRUCT, self.info, adpcminforef.offset + offset)
					adpcminfo = DSPADPCMInfo(rawentry)
					self.channelinfo.append(adpcminfo)
				elif self.codec == IMAADPCM:
					rawentry = self.unpack_from(BCSTM_IMAADPCM_INFO_STRUCT, self.info, adpcminforef.offset + offset)
					adpcminfo = IMAADPCMInfo(rawentry)
					self.channelinfo.append(adpcminfo)
				else:
					self.channelinfo.append(None)
	
	def read_streaminfo(self, data):
		self.codec = data[0]
		self.islooping = bool(data[1])
		self.channel_count = data[2]
		self.sample_rate = data[4]
		self.loop_start = data[5]
		self.loop_end = data[6]
		self.sampleblock_count = data[7]
		self.sampleblock_size = data[8]
		self.sampleblock_samplecount = data[9]
		self.last_sampleblock_size = data[10]
		self.last_sampleblock_samplecount = data[11]
		self.last_sampleblock_paddedsize = data[12]
		self.seek_datasize = data[13]
		self.seek_interval_samplecount = data[14]
		self.sampledata_ref = Reference(data[15])
		if self.verbose:
			print('Codec: %s' % CODECS[self.codec])
			print('Channel count: %d' % self.channel_count)
			print('Sample rate: %d' % self.sample_rate)
			print('Looping: %s' % self.islooping)
			print('Loop: %d - %d' % (self.loop_start, self.loop_end))
	
	def readSEEK(self):
		NotImplemented
	
	def clamp(self, value):
		if value < -32767:
			return -32767
		elif value > 32767:
			return 32767
		else:
			return value
	
	def readDATA(self):
		#self.wav = WAV.new(rate=self.sample_rate, channels=self.channel_count)
		data = self.data[self.sampledata_ref.offset + 8:]  #Strips the magic
		ptr = 0
		channels = [[] for i in range(self.channel_count)]
		for i in range(self.sampleblock_count):
			for c in range(self.channel_count):
				block = data[ptr: ptr + self.sampleblock_size]
				ptr += self.sampleblock_size
				channels[c] += self.decode_block(block, self.channelinfo[c], i == self.sampleblock_count - 1)
		if len(self.trackinfo) > 0:
			for i, info in enumerate(self.trackinfo):
				track = [channels[j] for j in info.channelindex]
				wav = WAV.new(rate=self.sample_rate, channels=len(info.channelindex))
				wav.samples = track
				outname = self.outname + '_track%d' % i
				wav.save(make_outfile(outname, 'wav'))
		else:
			wav = WAV.new(rate=self.sample_rate, channels=self.channel_count)
			wav.samples = channels
			outname = self.outname
			wav.save(make_outfile(outname, 'wav'))
	
	def decode_block(self, block, info, last):
		#Inspirated from vgmstream
		hist1 = info.context.previous_sample
		hist2 = info.context.second_previous_sample
		sample = 0
		ptr = 0
		ret = []
		samplecount = self.last_sampleblock_samplecount if last else self.sampleblock_samplecount
		while len(ret) < samplecount:
			try:
				header, ptr = self.uint8(block, ptr)
			except:
				break
			scale, coef_index = self.nibbles(header)
			scale = 1 << scale
			coef1, coef2 = info.param[coef_index % 8]
			for i in range(7):
				byte, ptr = self.uint8(block, ptr)
				nibbles = self.signed_nibbles(byte)
				if i == 0:
					adpcm_nibble = nibbles[0]
				else:
					adpcm_nibble = nibbles[1]
				sample = self.clamp(((adpcm_nibble * scale) << 11) + 1024 + ((coef1 * hist1) + (coef2 * hist2)) >> 11)
				hist2 = hist1
				hist1 = sample
				ret.append(sample / 32768)
		return ret
