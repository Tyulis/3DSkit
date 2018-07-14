# -*- coding:utf-8 -*-

import numpy as np

DBL_EPSILON = np.finfo(np.double).eps
DBL_MAX = np.finfo(np.double).max
CLAMP = lambda val, low, high: (low if val < low else (high if val > high else val))


def innerProductMerge(vecOut, pcmBuf):
	for i in range(3):
		vecOut[i] = 0.0
		for x in range(14):
			vecOut[i] -= pcmBuf[x - i] * pcmBuf[x]
 
def outerProductMerge(mtxOut, pcmBuf):
	for x in range(1, 3):
		for y in range(1, 3):
			mtxOut[x, y] = 0.0
			for z in range(14):
				mtxOut[x, y] += pcmBuf[z - x] * pcmBuf[z - y]
 
def analyzeRanges(mtx, vecIdxsOut):
	recips = np.zeros(3, np.double)
 
	# Get greatest distance from zero
	for x in range(1, 3):
		val = max(np.abs(mtx[x, 1]), np.abs(mtx[x, 2]))
		if val < DBL_EPSILON:
			return True
		recips[x] = 1.0 / val
 
	maxIndex = 0
	for i in range(1, 3):
		for x in range(1, i):
			tmp = mtx[x, i]
			for y in range(1, x):
				tmp -= mtx[x, y] * mtx[y, i]
			mtx[x, i] = tmp
 
		val = 0.0
		for x in range(i, 3):
			tmp = mtx[x, i]
			for y in range(i, 3):
				tmp -= mtx[x, y] * mtx[y, i]
 
			mtx[x, i] = tmp
			tmp = np.abs(tmp) * recips[x]
			if tmp >= val:
				val = tmp
				maxIndex = x
		
		if maxIndex != i:
			for y in range(1, 3):
				tmp = mtx[maxIndex, y]
				mtx[maxIndex, y] = mtx[i, y]
				mtx[i, y] = tmp
			recips[maxIndex] = recips[i]
 
		vecIdxsOut[i] = maxIndex
 
		if mtx[i, i] == 0.0:
			return True
		if i != 2:
			tmp = 1.0 / mtx[i, i]
			for x in range(i + 1, 3):
				mtx[x, i] *= tmp
	
	# Get range
	min = 1.0e10
	max = 0.0
	for i in range(1, 3):
		tmp = np.abs(mtx[i, i])
		if tmp < min:
			min = tmp
		if tmp > max:
			max = tmp
 
	if min / max < 1.0e-10:
		return True
	return False
 
def bidirectionalFilter(mtx, vecIdxs, vecOut):
	x = 0
	for i in range(1, 3):
		index = vecIdxs[i]
		tmp = vecOut[index]
		vecOut[index] = vecOut[i]
		if x != 0:
			for y in range(x, i):
				tmp -= vecOut[y] * mtx[i, y]
		elif tmp != 0.0:
			x = i
		vecOut[i] = tmp
 
	for i in range(2, 0, -1):
		tmp = vecOut[i]
		for y in range(i + 1, 3):
			tmp -= vecOut[y] * mtx[i, y]
		vecOut[i] = tmp / mtx[i, i]
 
	vecOut[0] = 1.0
 
def quadraticMerge(inOutVec):
	v2 = inOutVec[2]
	tmp = 1.0 - (v2 * v2)
 
	if tmp == 0.0:
		return True
 
	v0 = (inOutVec[0] - (v2 * v2)) / tmp
	v1 = (inOutVec[1] - (inOutVec[1] * v2)) / tmp
 
	inOutVec[0] = v0
	inOutVec[1] = v1
 
	return np.abs(v1) > 1.0
 
def finishRecord(input, out):
	for z in range(1, 3):
		if input[z] >= 1.0:
			input[z] = 0.9999999999
		elif input[z] <= -1.0:
			input[z] = -0.9999999999
	out[0] = 1.0
	out[1] = (input[2] * input[1]) + input[1]
	out[2] = input[2]
 
def matrixFilter(src, dst):
	mtx = np.zeros((3, 3), dtype=np.double)
	mtx[2, 0] = 1.0
	for i in range(1, 3):
		mtx[2, i] = -src[i]
 
	for i in range(2, 0, -1):
		val = 1.0 - (mtx[i, i] * mtx[i, i])
		for y in range(1, i + 1):
			mtx[i - 1, y] = ((mtx[i, i] * mtx[i, y]) + mtx[i, y]) / val
 
	dst[0] = 1.0
	for i in range(1, 3):
		dst[i] = 0.0
		for y in range(1, i + 1):
			dst[i] += mtx[i, y] * dst[i - y]
 
def mergeFinishRecord(src, dst):
	tmp = np.zeros(3, dtype=np.double)
	val = src[0]
 
	dst[0] = 1.0
	for i in range(1, 3):
		v2 = 0.0
		for y in range(1, i):
			v2 += dst[y] * src[i - y]
		if val > 0.0:
			dst[i] = -(v2 + src[i]) / val
		else:
			dst[i] = 0.0
		tmp[i] = dst[i]
 
		for y in range(1, i):
			dst[y] += dst[i] * dst[i - y]
		val *= 1.0 - (dst[i] * dst[i])
	finishRecord(tmp, dst)
 
def contrastVectors(source1, source2):
	val = (source2[2] * source2[1] + -source2[1]) / (1.0 - source2[2] * source2[2])
	val1 = (source1[0] * source1[0]) + (source1[1] * source1[1]) + (source1[2] * source1[2])
	val2 = (source1[0] * source1[1]) + (source1[1] * source1[2])
	val3 = source1[0] * source1[2]
	return val1 + (2.0 * val * val2) + (2.0 * (-source2[1] * val + -source2[2]) * val3)

def filterRecords(vecBest, exp, records, recordCount):
	bufferList = np.zeros((8, 3), dtype=np.double)
 
	buffer1 = np.zeros(8, dtype=np.int)
	buffer2 = np.zeros(3, dtype=np.double)
 
	tempVal = 0
 
	for x in range(0, 2):
		for y in range(0, exp):
			buffer1[y] = 0
			for i in range(0, 3):
				bufferList[y, i] = 0.0
		for z in range(0, recordCount):
			index = 0
			value = 1.0e30
			for i in range(0, exp):
				tempVal = contrastVectors(vecBest[i], records[z])
				if tempVal < value:
					value = tempVal
					index = i
			buffer1[index] += 1
			matrixFilter(records[z], buffer2)
			for i in range(0, 3):
				bufferList[index, i] += buffer2[i]
 
		for i in range(0, exp):
			if buffer1[i] > 0:
				for y in range(0, 3):
					bufferList[i, y] /= buffer1[i]
 
		for i in range(0, exp):
			mergeFinishRecord(bufferList[i], vecBest[i])

def generateDSPADPCMcoefs(coefsOut, source, samples):
	numFrames = int((samples + 13) / 14)
 
	blockBuffer = np.zeros(0x3800, dtype=np.int16)
	pcmHistBuffer = np.zeros((2, 14), dtype=np.int16)
 
	vec1 = np.zeros(3, dtype=np.double)
	vec2 = np.zeros(3, dtype=np.double)
 
	mtx = np.zeros((3, 3), dtype=np.double)
	vecIdxs = np.zeros(3, dtype=np.int)
 
	records = np.zeros((numFrames * 2, 3), dtype=np.double)
	recordCount = 0
 
	vecBest = np.zeros((8, 3), dtype=np.double)
 
	# Iterate though 1024-block frames
	sourcepos = 0
	x = samples
	while x > 0:
		if x > 0x3800:  # Full 1024-block frame
			frameSamples = 0x3800
			x -= 0x3800
		else:  # Partial frame
			# Zero lingering block samples
			frameSamples = x
			for z in range(0, 14):
				if z + frameSamples < 0x3800:
					break
				blockBuffer[frameSamples + z] = 0
			x = 0
 
		# Copy (potentially non-frame-aligned PCM samples into aligned buffer)
		blockBuffer[0: frameSamples] = source[sourcepos: sourcepos + frameSamples]
		sourcepos += frameSamples

		i = 0
		while i < frameSamples:
			for z in range(0, 14):
				pcmHistBuffer[0, z] = pcmHistBuffer[1, z]
			for z in range(0, 14):
				pcmHistBuffer[1, z] = blockBuffer[i]
				i += 1
			innerProductMerge(vec1, pcmHistBuffer[1])
			if np.abs(vec1[0]) > 10.0:
				outerProductMerge(mtx, pcmHistBuffer[1])
				if not analyzeRanges(mtx, vecIdxs):
					bidirectionalFilter(mtx, vecIdxs, vec1)
					if not quadraticMerge(vec1):
						finishRecord(vec1, records[recordCount])
						recordCount += 1
	vec1[0] = 1.0
	vec1[1] = 0.0
	vec1[2] = 0.0
 
	for z in range(0, recordCount):
		matrixFilter(records[z], vecBest[0])
		for y in range(1, 3):
			vec1[y] += vecBest[0, y]
	for y in range(1, 3):
		vec1[y] /= recordCount
	mergeFinishRecord(vec1, vecBest[0])

	exp = 1
	w = 0
	while w < 3:
		vec2[0] = 0.0
		vec2[1] = -1.0
		vec2[2] = 0.0
		for i in range(0, exp):
			for y in range(0, 3):
				vecBest[exp + i, y] = (0.01 * vec2[y]) + vecBest[i, y]
		w += 1
		exp = 1 << w
		filterRecords(vecBest, exp, records, recordCount)
 
	# Write output
	for z in range(0, 8):
		d = -vecBest[z, 1] * 2048.0
		if d > 0.0:
			coefsOut[z * 2] = 32767 if d > 32767.0 else round(d)
		else:
			coefsOut[z * 2] = -32768 if d < -32768.0 else round(d)
 
		d = -vecBest[z, 2] * 2048.0
		if d > 0.0:
			coefsOut[z * 2 + 1] = 32767 if d > 32767.0 else round(d)
		else:
			coefsOut[z * 2 + 1] = -32768 if d < -32768.0 else round(d)

# FIXME: SUPER SLOW
def decodeDSPADPCMblock(adpcm, pcmout, coefs, samplecount, blockstart, last1, last2):
	framecount = samplecount // 14
	framestart = 0
	sampleidx = 0
	sample = 0
	for i in range(0, framecount):
		info = adpcm[framestart]
		shift = (info & 0x0f) + 11
		coefindex = info >> 4
		coef1 = coefs[coefindex * 2]
		coef2 = coefs[coefindex * 2 + 1]
		for byteidx in range(0, 7):
			byte = adpcm[framestart + byteidx + 1]
			high = byte >> 4
			low = byte & 0x0f
			if high > 7:
				high -= 16
			if low > 7:
				low -= 16
			sample = CLAMP(((high << shift) + coef1 * last1 + coef2 * last2 + 1024) >> 11, -32768, 32767)
			pcmout[blockstart + sampleidx] = sample
			last2 = last1
			last1 = sample
			sampleidx += 1
			sample = CLAMP(((low << shift) + coef1 * last1 + coef2 * last2 + 1024) >> 11, -32768, 32767)
			pcmout[blockstart + sampleidx] = sample
			last2 = last1
			last1 = sample
			sampleidx += 1
		framestart += 8
	return last1, last2

def encodeDSPADPCMchannel(coefs, pcm, adpcmout, seek, samplecount, blocksamplecount, channelidx, channelcount, loopstart):
	contexts = np.zeros(4, dtype=np.int)
	blockidx = channelidx
	outpos = 0
	pcmbuf = np.zeros(16, dtype=np.int16)
	insamples = np.zeros((8, 16), dtype=np.int16)
	outsamples = np.zeros((8, 14), dtype=np.int16)
	last1 = last2 = p
	bestidx = 0
	scale = np.zeros(8, dtype=np.int)
	distaccum = np.zeros(8, dtype=np.double)
	for framepos in range(0, samplecount, 14):
		if framepos % blocksamplecount == 0:
			seek[blockidx * 2] = last1
			seek[blockidx * 2 + 1] = last2
			blockidx += channelcount
		pcmbuf[0] = last2
		pcmbuf[1] = last1
		for j in range(0, 14):
			pcmbuf[j + 2] = pcm[framepos + j]
		for coefidx in range(0, 8):
			coef1 = coefs[coefidx * 2]
			coef2 = coefs[coefidx * 2 + 1]
			insamples[coefidx, 0] = last2
			insamples[coefidx, 1] = last1
			distance = 0
			for s in range(0, 14):
				insamples[coefidx, s + 2] = v1 = ((pcmbuf[s] * coef2) + (pcmbuf[s + 1] * coef1)) / 2048
				v2 = pcmbuf[s + 2] - v1
				v3 = CLAMP(v2, -32768, 32767)
				if np.abs(v3) > np.abs(distance):
					distance = v3
			scale[coefidx] = 0
			while scale[coefidx] <= 12 and ((distance > 7) or (distance < -8)):
				scale[coefidx] += 1
				distance /= 2
			scale[coefidx] = -1 if scale[coefidx] <= 1 else scale[coefidx] - 2
			loop = True
			while loop:
				scale[coefidx] += 1
				distaccum[coefidx] = 0
				index = 0
				for s in range(0, 14):
					v1 = ((insamples[coefidx, s] * coef2) + (insamples[coefidx, s + 1] * coef1))
					v2 = ((pcmbuf[s + 2] << 11) - v1) / 2048
					v3 = int((v2 / (1 << scale[coefidx]) + 0.4999999)) if v2 > 0 else int((v2 / (1 << scale[coefidx]) - 0.4999999))
					if v3 < -8:
						v3 = -8 - v3
						if index < v3:
							index = v3
						v3 = -8
					elif v3 > 7:
						v3 -= 7
						if index < v3:
							index = v3
						v3 = 7
					outsamples[coefidx, s] = v3
					v1 = (v1 + ((v3 * (1 << scale[coefidx])) << 11) + 1024) >> 11
					insamples[coefidx, s + 2] = v2 = CLAMP(v1, -32768, 32767)
					v3 = pcmbuf[s + 2] - v2
					distaccum[coefidx] += v3 * v3
				x = index + 8
				while x > 256:
					scale[coefidx] += 1
					if scale[coefidx] >= 12:
						scale[coefidx] = 11
					x >>= 1
				if (scale[coefidx] < 12) and (index > 1):
					break
		mindist = DBL_MAX
		for coefidx in range(0, 8):
			if distaccum[coefidx] < mindist:
				mindist = distaccum[coefidx]
				bestidx = coefidx
		if framepos == 0:
			contexts[0] = bestidx
			contexts[1] = scale[bestidx]
		elif loopstart >= framepos and loopstart < framepos + 14:
			contexts[2] = bestidx
			contexts[3] = scale[bestidx]
		adpcmout[outpos] = (bestidx << 4) | (scale[bestidx] & 0x0f)
		outpos += 1
		for j in range(0, 7):
			adpcmout[outpos] = (outsamples[bestidx, j * 2] << 4) | (outsamples[bestidx, j * 2 + 1] & 0x0f)
			outpos += 1
		last1 = insamples[bestidx, 15]
		last2 = insamples[bestidx, 14]
	return contexts