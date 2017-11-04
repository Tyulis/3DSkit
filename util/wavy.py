# -*- coding:utf-8 -*-
import struct

PCM=1


class WAV (object):
	@classmethod
	def new(cls, rate=44100, channels=2, format=PCM, bitspersample=16):
		ins = cls()
		ins.samples = [[] for i in range(channels)]
		ins.channels = channels
		ins.fech = rate
		ins.format = format
		ins.bitspersample = bitspersample
		return ins
	
	@classmethod
	def load(cls, filename):
		ins = cls()
		f = bread(filename)
		hdr = f[0:44]
		data = f[44:]
		hdata = struct.unpack('<4sI4s4sIHHIIHH4sI', hdr)
		ins.format = hdata[5]
		ins.channels = hdata[6]
		ins.fech = hdata[7]
		ins.bitspersample = hdata[10]
		s = '<%d%s' % (ins.channels, ('B' if ins.bitspersample == 8 else 'h'))
		rawsamples = [u for u in struct.iter_unpack(s, data)]
		if ins.bitspersample == 8:
			ins.samples = [[(s - 127) / 127 for s in sample] for sample in rawsamples]
		elif ins.bitspersample == 16:
			ins.samples = [[s / 32767 for s in sample] for sample in rawsamples]
		return ins
	
	def _writeheader(self):
		bytepersample = self.bitspersample // 8
		byteperbloc = bytepersample * self.channels
		hdr = struct.pack('<4sI4s4sIHHIIHH4sI', 
			b'RIFF', len(self.samples) * byteperbloc + 32,
			b'WAVE', b'fmt ', 0x10,
			self.format, self.channels, self.fech,
			byteperbloc * self.fech, byteperbloc,
			self.bitspersample, b'data',
			byteperbloc * len(self.samples)
		)
		self.file.write(hdr)
	
	def _serialize(self, samples):
		final = b''
		#Best optimisation ever. Totally unreadable.
		if self.bitspersample == 8:
			final = b''.join([b''.join([struct.pack('<B', int(e * 127 + 127)) for e in s]) for s in samples])
		elif self.bitspersample == 16:
			final = b''.join([b''.join([struct.pack('<h', int(e * 32767)) for e in s]) for s in samples])
		else:
			raise ValueError('Not a supported bits per sample count')
		return final
	
	def save(self, filename):
		self.file = open(filename, 'wb')
		self._writeheader()
		data = self._serialize(self.samples)
		self.file.write(data)
		self.file.close()
		return 0
