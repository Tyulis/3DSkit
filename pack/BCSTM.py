# -*- coding:utf-8 -*-
import wave
import math
import time
import numpy as np
from util import error, BOMS
from util.utils import ClsFunc
from util.rawutil import TypeWriter
from util.filesystem import *

try:
	import c3DSkit
except:
	c3DSkit = None

NULL = 0xFFFFffff

PCM8 = 0
PCM16 = 1
DSPADPCM = 2
IMAADPCM = 3

CODECS = {
	'PCM8': PCM8,
	'PCM16': PCM16,
	'DSPADPCM': DSPADPCM,
	'IMAADPCM': IMAADPCM,
}

DOUBLE_EPSILON = np.finfo(np.double).eps

class DSPADPCMContext (object):
	def __init__(self):
		self.predictor = 0
		self.scale = 0
		self.last1 = 0
		self.last2 = 0
	
	def data(self):
		info = (self.predictor << 4) | self.scale
		return (info, 0, self.last1, self.last2)

class DSPADPCMChannelInfo (object):
	def __init__(self):
		self.param = np.zeros(16, dtype=np.int16)
		self.context = DSPADPCMContext()
		self.loopcontext = DSPADPCMContext()

class packBCSTM(ClsFunc, TypeWriter):
	def main(self, filenames, outname, endian, verbose, opts={}):
		self.byteorder = endian
		self.verbose = verbose
		if 'format' in opts:
			self.codec = CODECS[opts['format'].upper().replace('-', '')]
		else:
			self.codec = DSPADPCM
		if 'loop' in opts:
			self.loopstart, self.loopend = tuple(map(int, opts['loop'].split('-')))
			self.islooping = True
		else:
			self.loopstart = self.loopend = 0
			self.islooping = False
		if 'writetracks' in opts:
			self.writetracks = True if opts['writetrack'].lower() == 'true' else False
		else:
			self.writetracks = True
		self.loadchannels(filenames)
		self.pack_sampledata()
		self.out = open(outname, 'wb')
		self.packINFO()
		self.packSEEK()
		self.packDATA()
		self.pack_header()
		self.out.close()
	
	def loadchannels(self, filenames):
		self.channelcount = 0
		self.channels = []
		self.tracks = []
		self.trackcount = len(filenames)
		self.samplecount = None
		self.samplerate = None
		for name in filenames:
			if self.verbose:
				print('Loading audio data from %s' % name)
			wav = wave.open(name, 'rb')
			if self.samplecount is not None and self.samplecount != wav.getnframes():
				error.InvalidInputError('Track in file %s contains %d frames, expected %d (tracks should have the same frame count)' % (name, wav.getnframes(), self.samplecount))
			self.samplecount = wav.getnframes()
			if self.samplerate is not None and self.samplerate != wav.getframerate():
				error.InvalidInputError('Track in file %s has frame rate %d, expected %d (tracks should have the same frame rate)' % (name, wav.getframerate(), self.samplerate))
			self.samplerate = wav.getframerate()
			channelcount = wav.getnchannels()
			self.tracks.append(tuple(range(self.channelcount, self.channelcount + channelcount)))
			self.channelcount += channelcount
			audiodata = np.fromstring(wav.readframes(self.samplecount), dtype=np.int16)
			channels = [np.ascontiguousarray(audiodata[i::channelcount]) for i in range(channelcount)]
			self.channels.extend(channels)

	def pack_header(self):
		self.out.seek(0, 2)
		filesize = self.out.tell()
		self.out.seek(0)
		self.pack('>4sH', b'CSTM', BOMS[self.byteorder], self.out)
		self.pack('H2I2H', 0x40, 0x02000000, filesize, 3, 0, self.out)
		self.pack('2H2I', 0x4000, 0, self.infopos, self.infosize, self.out)
		self.pack('2H2I', 0x4001, 0, self.seekpos, self.seeksize, self.out)
		self.pack('2H2I', 0x4002, 0, self.datapos, self.datasize, self.out)
	
	def packINFO(self):
		self.out.seek(64)  #CSTM header is 56 bytes long
		self.infopos = self.out.tell()
		self.out.write(b'INFO____')  #____ will be replaced by the section size
		self.pack('2HI', 0x4100, 0, 0x18, self.out)  #Stream info ref
		if self.writetracks:
			self.pack('2HI', 0x0101, 0, 0x50, self.out)  #Track info ref table ref
			self.pack('2HI', 0x0101, 0, 0x50 + self.trackcount * 8 + 4, self.out)  #Channel info ref table ref
		else:
			self.pack('2HI', 0x0101, 0, NULL)
			self.pack('2HI', 0x0101, 0, 0x50)
		self.pack('4B11I 2HI', self.codec, self.islooping, self.channelcount, 0,
								self.samplerate, self.loopstart, self.loopend, self.blockcount + 1, self.blocksize,
								self.blocksamplecount, self.lastblocksize, self.lastblocksamplecount, self.lastblockpaddedsize,
								4, self.blocksamplecount, 0x1F00, 0, 0x18, self.out)
		if self.writetracks:
			tracktablepos = self.out.tell()
			self.pack('I', self.trackcount, self.out)
			entrypos = 4 + self.trackcount * 8 + 4 + self.channelcount * 8
			for i in range(self.trackcount):
				self.out.seek(tracktablepos + entrypos)
				startpos = self.out.tell()
				self.pack('2BH 2HI I(%dB)4a' % len(self.tracks[i]), 127, 64, 0, 0x0100, 0, 0x0c, len(self.tracks[i]), self.tracks[i], self.out)
				size = self.out.tell() - startpos
				self.out.seek(tracktablepos + 4 + i * 8)
				self.pack('2HI', 0x4101, 0, entrypos, self.out)
				entrypos += size
			endpos = tracktablepos + entrypos
		else:
			endpos = self.channelcount * 8 + 4
		self.out.seek(tracktablepos + 4 + self.trackcount * 8)
		tablepos = self.out.tell()
		entrypos = endpos - tablepos
		if self.codec == DSPADPCM:
			self.pack('I%d[2HI]' % self.channelcount, self.channelcount, [(0x4102, 0, entrypos + i * 8) for i in range(self.channelcount)], self.out)
			self.out.seek(endpos)
			for i in range(self.channelcount):
				self.pack('2HI', 0x0300, 0, (self.channelcount - i) * 8 + 46 * i, self.out)
			for i, info in enumerate(self.channelinfos):
				self.pack('(16h)(2B2h)(2B2h)H', tuple(info.param), info.context.data(), info.loopcontext.data(), 0, self.out)
		entrypos = endpos - self.out.tell()
		size = self.out.tell() - self.infopos
		padding = 0x20 - (size % 0x20 or 0x20)
		self.out.write(b'\x00' * padding)
		size += padding
		self.out.seek(self.infopos + 4)
		self.pack('I', size, self.out)
		self.infosize = size
	
	def packSEEK(self):
		self.out.seek(self.infopos + self.infosize)
		self.seekpos = self.out.tell()
		self.seeksize = 8 + 2 * len(self.seek)
		padding = 0x20 - (self.seeksize % 0x20 or 0x20)
		self.seeksize += padding
		self.pack('4sI', b'SEEK', self.seeksize, self.out)
		self.out.write(self.seek.tostring())
		self.out.write(b'\x00' * padding)
	
	def pack_sampledata(self):
		if self.codec == DSPADPCM:
			if c3DSkit is not None:
				self.packDSPADPCM_c3DSkit()
			else:
				self.packDSPADPCM_py3DSkit()
		else:
			error.NotImplementedError('This codec is not yet implemented')
	
	def packDSPADPCM_c3DSkit(self):
		self.blocksamplecount = 14336
		self.blocksize = (self.blocksamplecount // 14) * 8
		self.blockcount = self.samplecount // self.blocksamplecount
		self.lastblocksamplecount = self.samplecount - self.blocksamplecount * self.blockcount
		self.lastblocksize = (self.lastblocksamplecount // 14) * 8
		self.lastblockpaddedsize = self.lastblocksize + (0x20 - (self.lastblocksize % 0x20 or 0x20))
		self.seeksize = (self.blockcount + 1) * self.channelcount * 2
		self.seek = np.ascontiguousarray(np.zeros(self.seeksize, dtype = np.int16))
		self.channelinfos = [DSPADPCMChannelInfo() for chan in self.channels]
		channeloutsize = (self.samplecount // 14) * 8
		self.channelouts = [np.ascontiguousarray(np.zeros(channeloutsize, dtype=np.uint8)) for chan in self.channels]
		for i, (info, channel, out) in enumerate(zip(self.channelinfos, self.channels, self.channelouts)):
			if self.verbose:
				print('Calculating coefficients for channel %d' % (i + 1))
			c3DSkit.generateDSPADPCMcoefs(info.param, channel, self.samplecount)
			if self.verbose:
				print('Encoding channel %d' % (i + 1))
			contexts = c3DSkit.encodeDSPADPCMchannel(info.param, channel, out, self.seek, self.samplecount, self.blocksamplecount, i, self.channelcount, self.loopstart)
			info.loopcontext.last1 = channel[self.loopstart - 1]
			info.loopcontext.last2 = channel[self.loopstart - 2]
			info.context.predictor, info.context.scale, info.loopcontext.predictor, info.loopcontext.scale = contexts
	
	def packDSPADPCM_py3DSkit(self):
		error.UnsupportedSettingError('You need c3DSkit installed to encode DSP-ADPCM streams')
	
	def packDATA(self):
		self.out.seek(self.seekpos + self.seeksize)
		self.datapos = self.out.tell()
		self.datasize = (self.samplecount // 14) * 8 * self.channelcount + 8
		self.pack('4sI', b'DATA', self.datasize, self.out)
		self.out.write(0x18 * b'\x00')
		pos = 0
		bs = self.out.tell()
		for i in range(self.blockcount):
			for out in self.channelouts:
				self.out.write(out[pos: pos + self.blocksize])
			pos += self.blocksize
		for out in self.channelouts:
			self.out.write(out[pos:])
