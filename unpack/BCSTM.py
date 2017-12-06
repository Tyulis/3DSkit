# -*- coding:utf-8 -*-
import wave
import numpy as np
from util import error
import util.rawutil as rawutil
from util.fileops import *
from util.funcops import ClsFunc, byterepr

rawutil.register_sub('S', '(2H2I)')  #sized refs
rawutil.register_sub('R', '(2HI)')  #references
rawutil.register_sub('T', 'I/p1[2HI]')  #ref table

BCSTM_HEADER_STRUCT = '4s2H 2I2H SSS'
BCSTM_INFO_STRUCT = '4sI RRR (4B11IR) $'
BCSTM_SEEK_STRUCT = '4sI {H}'
BCSTM_DATA_STRUCT = '4sI 24s $'
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
		self.channelindex = [el[0] for el in data[5]]


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


class DSPADPCMDecoder (object):
	def __init__(self, info:'DSPADPCMInfo'):
		self.info = info
	
	def updatelast(self, last1, last2):
		self.last1 = last1
		self.last2 = last2
	
	def getdata(self, offset, length, samplecount):
		out = np.zeros(samplecount, dtype=np.int16)
		sampleidx = 0
		for i in range(offset + 1, offset + length, 8):
			scale = 1 << (self.data[i - 1] & 0x0f)
			coef = self.data[i - 1] >> 4
			coef1, coef2 = self.info.param[coef]
			for j in range(7):
				if sampleidx >= samplecount:
					break
				high = self.data[i + j] >> 4
				low = self.data[i + j] & 0x0f
				if high >= 8:
					high -= 16
				if low >= 8:
					low -= 16
				val = (((high * scale) << 11) + 1024.0 + (coef1 * self.last1 + coef2 * self.last2)) / 2048.0
				sample = self.clamp(val)
				out[sampleidx] = sample
				sampleidx += 1
				if sampleidx >= samplecount:
					break
				self.last2 = self.last1
				self.last1 = val
				val = (((low * scale) << 11) + 1024.0 + (coef1 * self.last1 + coef2 * self.last2)) / 2048.0
				sample = self.clamp(val)
				out[sampleidx] = sample
				sampleidx += 1
				self.last2 = self.last1
				self.last1 = val
		return out
				
	
	def clamp(self, val):
		if val < -32768:
			return -32768
		elif val > 32767:
			return 32767
		else:
			return int(val)


class extractBCSTM (ClsFunc, rawutil.TypeReader):
	def main(self, filename, data, verbose, opts={}):
		self.verbose = verbose
		self.filename = filename
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
		magic, length, samples = self.unpack(BCSTM_SEEK_STRUCT, self.seek)
		if magic != b'SEEK':
			error('Bad SEEK magic %s' % byterepr(magic), 301)
		self.seek_samples = [el[0] for el in samples]
	
	def readDATA(self):
		magic, length, padding, self.audiodata = self.unpack(BCSTM_DATA_STRUCT, self.data)
		self.samplecount = (len(self.audiodata) // (self.sampleblock_size * self.channel_count)) * self.sampleblock_samplecount + self.last_sampleblock_samplecount
		if magic != b'DATA':
			error('Bad DATA magic %s' % byterepr(magic), 301)
		self.channels = []
		if self.codec == DSPADPCM:
			DSPADPCMDecoder.data = self.audiodata  #Avoid multiple transmissions of this
		for channum in range(self.channel_count):
			if self.verbose:
				print('Reading channel %d' % (channum + 1))
			if self.codec == PCM16:
				self.channels.append(self.extractPCM16channel(channum))
			elif self.codec == DSPADPCM:
				self.channels.append(self.extractDSPADPCMchannel(channum))
		if len(self.trackinfo) > 0:
			for i, info in enumerate(self.trackinfo):
				if self.verbose:
					print('Extracting track %d' % (i + 1))
				self.extract_track(info.channelindex)
		else:
			if self.verbose:
				print('Extracting the unique track')
			self.extract_track(tuple(range(len(self.channels))))
		
	def extract_track(self, channelindex):
		#samples = np.array([[self.channels[i][j] for j in range(self.samplecount)] for i in channelindex], dtype=np.int16)
		samples = np.array(tuple(zip(*[self.channels[i] for i in channelindex])), dtype=np.int16)
		if len(self.trackinfo) > 1:
			trackname = os.path.splitext(self.filename)[0] + '_track%d.wav' % (i + 1)
		else:
			trackname = os.path.splitext(self.filename)[0] + '.wav'
		wav = wave.open(trackname, 'w')
		wav.setframerate(self.sample_rate)
		wav.setnchannels(len(channelindex))
		wav.setsampwidth(2)
		wav.writeframesraw(samples.tostring())
		wav.close()
	
	def extractPCM16channel(self, channum):
		#Inspirated from EveryFileExplorer
		samples = []
		for i in range(0, len(self.audiodata), self.sampleblock_size * self.channel_count):
			if i + self.sampleblock_size * self.channel_count < self.audiodata:
				for j in range(0, self.sampleblock_size, 2):
					pos = i + channum * self.sampleblock_size + j
					samples.append(self.uint16(self.audiodata, pos)[0])
			elif self.sampleblock_size != self.last_sampleblock_size:
				for j in range(0, self.last_sampleblock_size, 2):
					pos = i + channum * self.last_sampleblock_paddedsize + j
					samples.append(self.uint16(self.audiodata, pos)[0])
		return samples
	
	def extractDSPADPCMchannel(self, channum):
		decoder = DSPADPCMDecoder(self.channelinfo[channum])
		samples = np.zeros(self.samplecount, dtype=np.int16)
		sampleidx = 0
		for i in range(0, len(self.audiodata), self.sampleblock_size * self.channel_count):
			prev2 = self.seek_samples[int((i / self.sampleblock_size) * 2 + channum * 2 + 1)]
			prev1 = self.seek_samples[int((i / self.sampleblock_size) * 2 + channum * 2)]
			if i != 0:
				samples[sampleidx - 2] = prev2
				samples[sampleidx- 1] = prev1
			decoder.updatelast(prev1, prev2);
			if i + self.sampleblock_size * self.channel_count < len(self.audiodata):
				samples[sampleidx: sampleidx + self.sampleblock_samplecount] = decoder.getdata(i + channum * self.sampleblock_size, self.sampleblock_size, self.sampleblock_samplecount)
				sampleidx += self.sampleblock_samplecount
			elif self.sampleblock_size != self.last_sampleblock_paddedsize:
				samples[sampleidx:] = decoder.getdata(i + channum * self.last_sampleblock_paddedsize, self.last_sampleblock_paddedsize, self.last_sampleblock_samplecount);
			else:
				break
		return samples
