# -*- coding:utf-8 -*-
import os
import wave
import numpy as np
from io import BytesIO
from util import error, ENDIANS, libkit
from util.utils import byterepr, ClsFunc
from util.filesystem import *
import util.rawutil as rawutil

# Very similar to BCSTM, without SEEK

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
		self.param = np.ascontiguousarray(np.array(data[0], dtype=np.int16))
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


class extractBCWAV (rawutil.TypeReader, ClsFunc):
	def main(self, filename, data, verbose, opts={}):
		self.outbase = os.path.splitext(filename)[0]
		self.verbose = verbose
		self.read_header(data)
		self.readINFO(data)
		self.readDATA(data)
	
	def read_header(self, data):
		self.byteorder = ENDIANS[rawutil.unpack_from('>H', data, 4)[0]]
		header = self.unpack_from('4s2H 2I2H SS', data, 0, 'magic, bom, headerlen, version, filesize, seccount, reserved, inforef, dataref')
		if header.magic != b'CWAV':
			error.InvalidMagicError('Invalid magic %s, expected CWAV' % byterepr(header.magic))
		self.filesize = header.filesize
		self.inforef = SizedRef(header.inforef)
		self.dataref = SizedRef(header.dataref)
	
	def readINFO(self, data):
		magic, size = self.unpack_from('4sI', data, self.inforef.offset)
		if magic != b'INFO':
			error.InvalidMagicError('Invalid INFO magic (got %s)' % byterepr(magic))
		self.codec, self.islooping, _, self.samplerate, self.loopstart, self.loopend, reserved = self.unpack_from('2BH 4I', data)
		if self.verbose:
			print('Codec: %s' % CODECNAMES[self.codec])
			print('Sample rate: %d' % self.samplerate)
			print('Looping: %s' % self.islooping)
		if self.islooping:
			print('Loop: %d - %d' % (self.loopstart, self.loopend))
		if self.codec != DSPADPCM:
			error.NotImplementedError('Codec %s is not yet implemented' % CODECNAMES[self.codec])
		channelinforefs = []
		for item in self.unpack_from('I/p1[2HI]', data)[1]:
			ref = Reference(item)
			if ref.offset == NULL:
				break
			else:
				channelinforefs.append(ref)
		self.channelcount = len(channelinforefs)
		if self.verbose:
			print('Channel count: %d' % self.channelcount)
		self.channelrefs = []
		self.channelinfos = []
		for ref in channelinforefs:
			start = data.tell()
			#TODO: Other codecs support
			samplesref = Reference(self.unpack_from('R', data, start)[0])
			self.channelrefs.append(samplesref)
			print(hex(samplesref.offset))
			inforef = Reference(self.unpack_from('RI', data, start)[0])  # Last I is reserved
			if self.codec == DSPADPCM:
				self.channelinfos.append(DSPADPCMInfo(self.unpack_from('(16h)(2B2H)(2B2H)H', data, start + inforef.offset)))
	
	def readDATA(self, data):
		magic, length = self.unpack_from('4sI', data, self.dataref.offset)
		if magic != b'DATA':
			error.InvalidMagicError('Invalid DATA magic (got %s)' % byterepr(magic))
		if self.verbose:
			print('Extracting DATA')
		#TODO: Other codecs
		if self.codec == DSPADPCM:
			self.decodeDSPADPCM(data)
		self.extract_track(self.channels)
	
	def decodeDSPADPCM(self, data):
		start = self.dataref.offset + self.channelrefs[-1].offset
		data.seek(0, 2)
		size = data.tell()
		self.samplecount = int(size * 14 / 8)
		self.channels = [np.ascontiguousarray(np.zeros(self.samplecount, dtype=np.int16)) for i in range(self.channelcount)]
		for channelidx in range(self.channelcount):
			last2 = self.channelinfos[channelidx].context.previous
			last1 = self.channelinfos[channelidx].context.secondprevious
			samplesref = self.channelrefs[channelidx]
			data.seek(self.dataref.offset + samplesref.offset)
			if channelidx < self.channelcount - 1:
				nextref = self.channelrefs[channelidx + 1]
				size = nextref.offset - samplerefs.offset
				adpcm = np.ascontiguousarray(np.fromstring(data.read(size), dtype=np.uint8))
			else:
				adpcm = np.ascontiguousarray(np.fromstring(data.read(), dtype=np.uint8))
			pcmout = self.channels[channelidx]
			param = self.channelinfos[channelidx].param
			last1, last2 = libkit.decodeDSPADPCMblock(adpcm, pcmout, param, adpcm.shape[0], 0, last1, last2)
	
	def extract_track(self, channels):
		filename = self.outbase + '.wav'
		wav = wave.open(filename, 'wb')
		wav.setframerate(self.samplerate)
		wav.setnchannels(len(channels))
		wav.setsampwidth(2)
		samples = np.array(tuple(zip(*channels)), dtype=np.int16)
		wav.writeframesraw(samples.tostring())
		wav.close()
