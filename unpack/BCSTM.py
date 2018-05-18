# -*- coding:utf-8 -*-
import os
import wave
import numpy as np
from io import BytesIO
from util import error, ENDIANS
from util.utils import byterepr, ClsFunc
from util.filesystem import *
import util.rawutil as rawutil

try:
	import c3DSkit
except:
	c3DSkit = None

rawutil.register_sub("R", "(2HI)")
rawutil.register_sub("S", "(2H2I)")
rawutil.register_sub("T", "I/p1(R)")

class Reference (object):
	def __init__(self, data):
		self.type, _, self.offset = data
	
	def __add__(self, diff):
		return Reference((self.type, None, self.offset + diff))


class SizedRef (object):
	def __init__(self, data):
		self.type, _, self.offset, self.size = data
	
	def __add__(self, diff):
		return SizedRef((self.type, None, self.offset + diff, self.size))

class TrackInfo (object):
	def __init__(self, data):
		self.volume, self.pan, _, tableref = data
		self.tableref = Reference(tableref)
		self.channels = None

class DSPADPCMContext (object):
	def __init__(self, data):
		self.predictor = data[0] >> 4
		self.scale = data[0] & 0x0f
		self.previous = data[2]
		self.secondprevious = data[3]

class DSPADPCMInfo (object):
	def __init__(self, data):
		self.param = np.array(data[0], dtype=np.int16)
		self.context = DSPADPCMContext(data[1])
		self.loopcontext = DSPADPCMContext(data[2])

NULL = 0xFFFFffff

PCM8 = 0
PCM16 = 1
DSPADPCM = 2
IMAADPCM = 3

CODECNAMES = {
	PCM8: "PCM8",
	PCM16: "PCM16",
	DSPADPCM: "DSP-ADPCM",
	IMAADPCM: "IMA-ADPCM",
}


class extractBCSTM (rawutil.TypeReader, ClsFunc):
	def main(self, filename, data, verbose, opts={}):
		self.outbase = os.path.splitext(filename)[0]
		self.verbose = verbose
		self.read_header(data)
		self.readINFO(data)
		self.readSEEK(data)
		self.readDATA(data)
	
	def read_header(self, data):
		self.byteorder = ENDIANS[rawutil.unpack_from(">H", data, 4)[0]]
		header = self.unpack_from("4s2H 2I2H SSS", data, 0, "magic, bom, headerlen, version, filesize, seccount, reserved, inforef, seekref, dataref")
		if header.magic != b"CSTM":
			error.InvalidMagicError("Invalid magic %s, expected CSTM" % byterepr(header.magic))
		self.filesize = header.filesize
		self.inforef = SizedRef(header.inforef)
		self.seekref = SizedRef(header.seekref)
		self.dataref = SizedRef(header.dataref)
	
	def readINFO(self, data):
		magic, size = self.unpack_from('4sI', data, self.inforef.offset)
		if magic != b'INFO':
			error.InvalidMagicError('Invalid INFO magic (got %s)' % byterepr(magic))
		streaminforef = Reference(self.unpack_from('R', data)[0])
		trackinforeftable = Reference(self.unpack_from('R', data)[0])
		channelinforeftable = Reference(self.unpack_from('R', data)[0])
		self.read_streaminfo(data)
		trackinforefs = []
		if trackinforeftable.offset != NULL:
			for item in self.unpack_from('I/p1[2HI]', data, self.inforef.offset + trackinforeftable.offset + 8)[1]:
				ref = Reference(item)
				if ref.offset == NULL:
					break
				else:
					trackinforefs.append(ref)
		channelinforefs = []
		if channelinforeftable.offset != NULL:
			for item in self.unpack_from('I/p1[2HI]', data, self.inforef.offset + channelinforeftable.offset + 8)[1]:
				ref = Reference(item)
				if ref.offset == NULL:
					break
				else:
					channelinforefs.append(ref)
		self.tracks = []
		for ref in trackinforefs:
			start = self.inforef.offset + trackinforeftable.offset + 8 + ref.offset
			track = TrackInfo(self.unpack_from('2BHR', data, start))
			track.channels = self.unpack_from('I/p1(B)', data, start + track.tableref.offset)[1]
			self.tracks.append(track)
		self.channelinfos = []
		for ref in channelinforefs:
			start = self.inforef.offset + channelinforeftable.offset + 8 + ref.offset
			#TODO: Other codecs support
			inforef = Reference(self.unpack_from('R', data, start)[0])
			if self.codec == DSPADPCM:
				self.channelinfos.append(DSPADPCMInfo(self.unpack_from('(16h)(2B2H)(2B2H)H', data, start + inforef.offset)))
	
	def read_streaminfo(self, data):
		self.codec, self.islooping, self.channelcount, _, self.samplerate, self.loopstart, self.loopend, self.blockcount, self.blocksize, self.blocksamplecount, self.lastblocksize, self.lastblocksamplecount, self.lastblockpaddedsize, self.seeksize, self.seekinterval, sampledataref = self.unpack_from("4B11I R", data)
		self.sampledataref = Reference(sampledataref)
		if self.codec != DSPADPCM:
			error.NotImplementedError('Codec %s is not yet implemented' % CODECNAMES[self.codec])
		if self.verbose:
			print('Codec: %s' % CODECNAMES[self.codec])
			print('Channel count: %d' % self.channelcount)
			print('Sample rate: %d' % self.samplerate)
			print('Looping: %s' % self.islooping)
		if self.islooping:
			print('Loop: %d - %d' % (self.loopstart, self.loopend))
	
	def readSEEK(self, data):
		if self.verbose:
			print('Reading SEEK')
		magic, length = self.unpack_from('4sI', data, self.seekref.offset)
		if magic != b'SEEK':
			error.InvalidMagicError('Invalid SEEK magic (got %s)' % byterepr(magic))
		self.seek = np.fromstring(data.read(length - 8), dtype=np.int16)
	
	def readDATA(self, data):
		magic, length = self.unpack_from('4sI', data, self.dataref.offset)
		if magic != b'DATA':
			error.InvalidMagicError('Invalid DATA magic (got %s)' % byterepr(magic))
		if self.verbose:
			print('Extracting DATA')
		self.blockcount -= 1
		data.seek(self.sampledataref.offset, 1)
		self.samplecount = self.blockcount * self.blocksamplecount + self.lastblocksamplecount
		self.channels = [np.zeros(self.samplecount, dtype=np.int16) for i in range(self.channelcount)]
		#TODO: Other codecs
		if self.codec == DSPADPCM:
			self.decodeDSPADPCM(data)
		if len(self.tracks) > 0:
			for i, track in enumerate(self.tracks):
				self.extract_track([self.channels[idx] for idx in track.channels], i)
		else:
			self.extract_track(self.channels)
	
	def decodeDSPADPCM(self, data):
		if c3DSkit is None:
			self.decodeDSPADPCM_py3DSkit(data)
		else:
			self.decodeDSPADPCM_c3DSkit(data)
	
	def decodeDSPADPCM_py3DSkit(self, data):
		curchannel = -1
		curblock = -1
		for blockidx in range(self.blockcount * self.channelcount):
			curchannel = (curchannel + 1) % self.channelcount
			if curchannel == 0:
				curblock += 1
				blockstart = curblock * self.blocksamplecount
				if self.verbose:
					print('Reading block %d' % curblock)
			sampleidx = 0
			last2 = self.seek[blockidx * 2 + 1]
			last1 = self.seek[blockidx * 2]
			buf = data.read(self.blocksize)
			# A bit of caching
			channel = self.channels[curchannel]
			param = self.channelinfos[curchannel].param
			infos = buf[::8]
			sampledatablocks = tuple(buf[i + 1: i + 8] for i in range(0, self.blocksize, 8))
			for info, sampledata in zip(infos, sampledatablocks):
				shift = (info & 0x0f) + 11
				coef1, coef2 = param[(info >> 4) * 2], param[(info >> 4) * 2 + 1]
				for nibbles in sampledata:
					high = nibbles >> 4
					low = nibbles & 0x0f
					if high >= 8:
						high -= 16
					if low >= 8:
						low -= 16
					sample = ((high << shift) + coef1 * last1 + coef2 * last2) / 2048
					channel[blockstart + sampleidx] = sample
					sample2 = ((low << shift) + coef1 * sample + coef2 * last1) / 2048
					channel[blockstart + sampleidx + 1] = sample2
					sampleidx += 2
					last2 = sample
					last1 = sample2
		curblock += 1
		blockstart = curblock * self.blocksamplecount
		for curchannel in range(self.channelcount):
			blockidx += 1
			sampleidx = 0
			last2 = self.seek[blockidx * 2 + 1]
			last1 = self.seek[blockidx * 2]
			buf = data.read(self.lastblockpaddedsize)
			# A bit of caching
			channel = self.channels[curchannel]
			param = self.channelinfos[curchannel].param
			infos = buf[::8]
			sampledatablocks = tuple(buf[i + 1: i + 8] for i in range(0, self.lastblocksize, 8))
			for info, sampledata in zip(infos, sampledatablocks):
				shift = (info & 0x0f) + 11
				coef1, coef2 = param[(info >> 4) * 2], param[(info >> 4) * 2 + 1]
				for nibbles in sampledata:
					if sampleidx >= self.lastblocksamplecount:
						break
					high = nibbles >> 4
					low = nibbles & 0x0f
					if high >= 8:
						high -= 16
					if low >= 8:
						low -= 16
					sample = ((high << shift) + coef1 * last1 + coef2 * last2) / 2048
					channel[blockstart + sampleidx] = sample
					if sampleidx + 1 >= self.lastblocksamplecount:
						break
					sample2 = ((low << shift) + coef1 * sample + coef2 * last1) / 2048
					channel[blockstart + sampleidx + 1] = sample2
					sampleidx += 2
					last2 = sample
					last1 = sample2
	
	def decodeDSPADPCM_c3DSkit(self, data):
		curchannel = -1
		curblock = -1
		for blockidx in range(self.blockcount * self.channelcount):
			curchannel = (curchannel + 1) % self.channelcount
			if curchannel == 0:
				curblock += 1
				blockstart = curblock * self.blocksamplecount
			last2 = self.seek[blockidx * 2 + 1]
			last1 = self.seek[blockidx * 2]
			adpcm = np.fromstring(data.read(self.blocksize), dtype=np.uint8)
			pcmout = self.channels[curchannel]
			param = self.channelinfos[curchannel].param
			last1, last2 = c3DSkit.decodeDSPADPCMblock(adpcm, pcmout, param, self.blocksamplecount, blockstart, last1, last2)
		curblock += 1
		blockstart = curblock * self.blocksamplecount
		for curchannel in range(self.channelcount):
			last2 = self.seek[blockidx * 2 + 1]
			last1 = self.seek[blockidx * 2]
			adpcm = np.fromstring(data.read(self.lastblockpaddedsize), dtype=np.uint8)
			pcmout = self.channels[curchannel]
			param = self.channelinfos[curchannel].param
			last1, last2 = c3DSkit.decodeDSPADPCMblock(adpcm, pcmout, param, self.lastblocksamplecount, blockstart, last1, last2)
	
	def extract_track(self, channels, index=None):
		if self.verbose:
			print('Extracting track %d' % (index + 1 if index is not None else 1))
		if index is None or len(self.tracks) <= 1:
			filename = self.outbase + '.wav'
		else:
			filename = self.outbase + '_track%d.wav' % index
		wav = wave.open(filename, 'wb')
		wav.setframerate(self.samplerate)
		wav.setnchannels(len(channels))
		wav.setsampwidth(2)
		samples = np.array(tuple(zip(*channels)), dtype=np.uint8)
		wav.writeframesraw(samples.tostring())
		wav.close()
