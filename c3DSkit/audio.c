#include "audio.h"

#define ABS(val) ((val < 0) ? -val : val)
#define LOG2(val) (log(val) / log(2))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define CLAMP(val, low, high) ((val < low) ? low : ((val > high) ? high : val))
 
// Most of the DSP-ADPCM encoding stuff is taken from github.com/jackoalan/gc-dspadpcm-encode
// With some adaptations to 3DSkit and new formats

typedef double tvec[3];
 
static inline void innerProductMerge(tvec vecOut, int16_t pcmBuf[14]){
	for (int i=0 ; i<=2 ; i++){
		vecOut[i] = 0.0f;
		for (int x=0 ; x<14 ; x++){
			vecOut[i] -= pcmBuf[x-i] * pcmBuf[x];
		}
	}
}
 
static inline void outerProductMerge(tvec mtxOut[3], int16_t pcmBuf[14]){
	for (int x=1 ; x<=2 ; x++){
		for (int y=1 ; y<=2 ; y++){
			mtxOut[x][y] = 0.0;
			for (int z=0 ; z<14 ; z++){
				mtxOut[x][y] += pcmBuf[z-x] * pcmBuf[z-y];
			}
		}
	}
}
 
static bool analyzeRanges(tvec mtx[3], int* vecIdxsOut){
	double recips[3];
	double val, tmp, min, max;
 
	/* Get greatest distance from zero */
	for (int x=1 ; x<=2 ; x++){
		val = MAX(fabs(mtx[x][1]), fabs(mtx[x][2]));
		if (val < DBL_EPSILON){
			return true;
		}
		recips[x] = 1.0 / val;
	}
 
	int maxIndex = 0;
	for (int i=1 ; i<=2 ; i++){
		for (int x=1 ; x<i ; x++){
			tmp = mtx[x][i];
			for (int y=1 ; y<x ; y++){
				tmp -= mtx[x][y] * mtx[y][i];
			}
			mtx[x][i] = tmp;
		}
 
		val = 0.0;
		for (int x=i ; x<=2 ; x++){
			tmp = mtx[x][i];
			for (int y=1 ; y<i ; y++){
				tmp -= mtx[x][y] * mtx[y][i];
			}
 
			mtx[x][i] = tmp;
			tmp = fabs(tmp) * recips[x];
			if (tmp >= val){
				val = tmp;
				maxIndex = x;
			}
		}
 
		if (maxIndex != i){
			for (int y=1 ; y<=2 ; y++){
				tmp = mtx[maxIndex][y];
				mtx[maxIndex][y] = mtx[i][y];
				mtx[i][y] = tmp;
			}
			recips[maxIndex] = recips[i];
		}
 
		vecIdxsOut[i] = maxIndex;
 
		if (mtx[i][i] == 0.0){
			return true;
		}
		if (i != 2){
			tmp = 1.0 / mtx[i][i];
			for (int x=i+1 ; x<=2 ; x++){
				mtx[x][i] *= tmp;
			}
		}
	}
 
	/* Get range */
	min = 1.0e10;
	max = 0.0;
	for (int i=1 ; i<=2 ; i++){
		tmp = fabs(mtx[i][i]);
		if (tmp < min){
			min = tmp;
		}
		if (tmp > max){
			max = tmp;
		}
	}
 
	if (min / max < 1.0e-10){
		return true;
	}
	return false;
}
 
static void bidirectionalFilter(tvec mtx[3], int* vecIdxs, tvec vecOut){
	double tmp;
	for (int i=1, x=0 ; i<=2 ; i++){
		int index = vecIdxs[i];
		tmp = vecOut[index];
		vecOut[index] = vecOut[i];
		if (x != 0){
			for (int y=x ; y<=i-1 ; y++){
				tmp -= vecOut[y] * mtx[i][y];
			}
		} else if (tmp != 0.0){
			x = i;
		}
		vecOut[i] = tmp;
	}
 
	for (int i=2 ; i>0 ; i--){
		tmp = vecOut[i];
		for (int y=i+1 ; y<=2 ; y++){
			tmp -= vecOut[y] * mtx[i][y];
		}
		vecOut[i] = tmp / mtx[i][i];
	}
 
	vecOut[0] = 1.0;
}
 
static bool quadraticMerge(tvec inOutVec){
	double v0, v1, v2 = inOutVec[2];
	double tmp = 1.0 - (v2 * v2);
 
	if (tmp == 0.0){
		return true;
	}
 
	v0 = (inOutVec[0] - (v2 * v2)) / tmp;
	v1 = (inOutVec[1] - (inOutVec[1] * v2)) / tmp;
 
	inOutVec[0] = v0;
	inOutVec[1] = v1;
 
	return fabs(v1) > 1.0;
}
 
static void finishRecord(tvec in, tvec out){
	for (int z=1 ; z<=2 ; z++){
		if (in[z] >= 1.0){
			in[z] = 0.9999999999;
		} else if (in[z] <= -1.0){
			in[z] = -0.9999999999;
		}
	}
	out[0] = 1.0;
	out[1] = (in[2] * in[1]) + in[1];
	out[2] = in[2];
}
 
static void matrixFilter(tvec src, tvec dst){
	tvec mtx[3];
	mtx[2][0] = 1.0;
	for (int i=1 ; i<=2 ; i++){
		mtx[2][i] = -src[i];
	}
 
	for (int i=2 ; i>0 ; i--){
		double val = 1.0 - (mtx[i][i] * mtx[i][i]);
		for (int y = 1; y <= i; y++){
			mtx[i-1][y] = ((mtx[i][i] * mtx[i][y]) + mtx[i][y]) / val;
		}
	}
 
	dst[0] = 1.0;
	for (int i=1 ; i<=2 ; i++){
		dst[i] = 0.0;
		for (int y=1 ; y<=i ; y++){
			dst[i] += mtx[i][y] * dst[i-y];
		}
	}
}
 
static void mergeFinishRecord(tvec src, tvec dst){
	tvec tmp;
	double val = src[0];
 
	dst[0] = 1.0;
	for (int i=1 ; i<=2 ; i++){
		double v2 = 0.0;
		for (int y=1 ; y<i ; y++){
			v2 += dst[y] * src[i-y];
		}
		if (val > 0.0){
			dst[i] = -(v2 + src[i]) / val;
		} else {
			dst[i] = 0.0;
		}
		tmp[i] = dst[i];
 
		for (int y=1 ; y<i ; y++){
			dst[y] += dst[i] * dst[i - y];
		}
		val *= 1.0 - (dst[i] * dst[i]);
	}
	finishRecord(tmp, dst);
}
 
static double contrastVectors(tvec source1, tvec source2){
	double val = (source2[2] * source2[1] + -source2[1]) / (1.0 - source2[2] * source2[2]);
	double val1 = (source1[0] * source1[0]) + (source1[1] * source1[1]) + (source1[2] * source1[2]);
	double val2 = (source1[0] * source1[1]) + (source1[1] * source1[2]);
	double val3 = source1[0] * source1[2];
	return val1 + (2.0 * val * val2) + (2.0 * (-source2[1] * val + -source2[2]) * val3);
}
 
static void filterRecords(tvec vecBest[8], int exp, tvec records[], int recordCount){
	tvec bufferList[8];
 
	int buffer1[8];
	tvec buffer2;
 
	int index;
	double value, tempVal = 0;
 
	for (int x=0 ; x<2 ; x++){
		for (int y=0 ; y<exp ; y++){
			buffer1[y] = 0;
			for (int i=0 ; i<=2 ; i++){
				bufferList[y][i] = 0.0;
			}
		}
		for (int z=0 ; z<recordCount ; z++){
			index = 0;
			value= 1.0e30;
			for (int i=0 ; i<exp ; i++){
				tempVal = contrastVectors(vecBest[i], records[z]);
				if (tempVal < value){
					value = tempVal;
					index = i;
				}
			}
			buffer1[index]++;
			matrixFilter(records[z], buffer2);
			for (int i=0 ; i<=2 ; i++){
				bufferList[index][i] += buffer2[i];
			}
		}
 
		for (int i=0 ; i<exp ; i++){
			if (buffer1[i] > 0){
				for (int y=0 ; y<=2 ; y++){
					bufferList[i][y] /= buffer1[i];
				}
			}
		}
 
		for (int i=0 ; i<exp ; i++){
			mergeFinishRecord(bufferList[i], vecBest[i]);
		}
	}
}
 
static void _generateDSPADPCMcoefs(int16_t* coefsOut, int16_t* source, int samples){
	int numFrames = (samples + 13) / 14;
	int frameSamples;
 
	int16_t* blockBuffer = (int16_t*)calloc(sizeof(int16_t), 0x3800);
	int16_t pcmHistBuffer[2][14] = {};
 
	tvec vec1;
	tvec vec2;
 
	tvec mtx[3];
	int vecIdxs[3];
 
	tvec* records = (tvec*)calloc(sizeof(tvec), numFrames * 2);
	int recordCount = 0;
 
	tvec vecBest[8];
 
	/* Iterate though 1024-block frames */
	for (int x=samples ; x>0 ;){
		if (x > 0x3800){ /* Full 1024-block frame */
			frameSamples = 0x3800;
			x -= 0x3800;
		} else {/* Partial frame */
			/* Zero lingering block samples */
			frameSamples = x;
			for (int z=0 ; z<14 && z+frameSamples<0x3800 ; z++){
				blockBuffer[frameSamples+z] = 0;
			}
			x = 0;
		}
 
		/* Copy (potentially non-frame-aligned PCM samples into aligned buffer) */
		memcpy(blockBuffer, source, frameSamples * sizeof(int16_t));
		source += frameSamples;
 
 
		for (int i=0 ; i<frameSamples ;){
			for (int z=0 ; z<14 ; z++){
				pcmHistBuffer[0][z] = pcmHistBuffer[1][z];
			}
			for (int z=0 ; z<14 ; z++){
				pcmHistBuffer[1][z] = blockBuffer[i++];
			}
			innerProductMerge(vec1, pcmHistBuffer[1]);
			if (fabs(vec1[0]) > 10.0){
				outerProductMerge(mtx, pcmHistBuffer[1]);
				if (!analyzeRanges(mtx, vecIdxs)){
					bidirectionalFilter(mtx, vecIdxs, vec1);
					if (!quadraticMerge(vec1)){
						finishRecord(vec1, records[recordCount]);
						recordCount++;
					}
				}
			}
		}
	}
 
	vec1[0] = 1.0;
	vec1[1] = 0.0;
	vec1[2] = 0.0;
 
	for (int z=0 ; z<recordCount ; z++){
		matrixFilter(records[z], vecBest[0]);
		for (int y=1 ; y<=2 ; y++){
			vec1[y] += vecBest[0][y];
		}
	}
	for (int y=1 ; y<=2 ; y++){
		vec1[y] /= recordCount;
	}
	mergeFinishRecord(vec1, vecBest[0]);

	int exp = 1;
	for (int w=0 ; w<3 ;){
		vec2[0] = 0.0;
		vec2[1] = -1.0;
		vec2[2] = 0.0;
		for (int i=0 ; i<exp ; i++){
			for (int y=0 ; y<=2 ; y++){
				vecBest[exp+i][y] = (0.01 * vec2[y]) + vecBest[i][y];
			}
		}
		++w;
		exp = 1 << w;
		filterRecords(vecBest, exp, records, recordCount);
	}
 
	/* Write output */
	for (int z=0 ; z<8 ; z++){
		double d;
		d = -vecBest[z][1] * 2048.0;
		if (d > 0.0){
			coefsOut[z * 2] = (d > 32767.0) ? (int16_t)32767 : (int16_t)lround(d);
		} else {
			coefsOut[z * 2] = (d < -32768.0) ? (int16_t)-32768 : (int16_t)lround(d);
		}
 
		d = -vecBest[z][2] * 2048.0;
		if (d > 0.0){
			coefsOut[z * 2 + 1] = (d > 32767.0) ? (int16_t)32767 : (int16_t)lround(d);
		} else {
			coefsOut[z * 2 + 1] = (d < -32768.0) ? (int16_t)-32768 : (int16_t)lround(d);
		}
	}
 
	/* Free memory */
	free(records);
	free(blockBuffer);
}

static int* _decodeDSPADPCMblock(uint8_t* adpcm, int16_t* pcmout, int16_t* coefs, int samplecount, int blockstart, int last1, int last2){
	int framecount = samplecount / 14;
	int framestart = 0;
	int sampleidx = 0;
	int16_t sample = 0;
	for (int i = 0; i < framecount; i++){
		uint8_t info = adpcm[framestart];
		int shift = (info & 0x0f) + 11;
		int coefindex = info >> 4;
		int16_t coef1 = coefs[coefindex * 2];
		int16_t coef2 = coefs[coefindex * 2 + 1];
		for (int byteidx = 0; byteidx < 7; byteidx++){
			uint8_t byte = adpcm[framestart + byteidx + 1];
			int high = byte >> 4;
			int low = byte & 0x0f;
			if (high > 7){
				high -= 16;
			}
			if (low > 7){
				low -= 16;
			}
			sample = CLAMP(((high << shift) + coef1 * last1 + coef2 * last2 + 1024) >> 11, -32768, 32767);
			pcmout[blockstart + sampleidx] = sample;
			last2 = last1;
			last1 = sample;
			sampleidx += 1;
			sample = CLAMP(((low << shift) + coef1 * last1 + coef2 * last2 + 1024) >> 11, -32768, 32767);
			pcmout[blockstart + sampleidx] = sample;
			last2 = last1;
			last1 = sample;
			sampleidx += 1;
		}
		framestart += 8;
	}
	int* lasts = (int*)calloc(2, sizeof(int));
	lasts[0] = last1;
	lasts[1] = last2;
	return lasts;
}

static int* _encodeDSPADPCMchannel(int16_t* coefs, int16_t* pcm, uint8_t* adpcmout, int16_t* seek, int samplecount, int blocksamplecount, int channelidx, int channelcount, int loopstart){
	int* contexts = (int*)calloc(4, sizeof(int));
	int blockidx = channelidx;
	uint8_t* outpos = adpcmout;
	int16_t pcmbuf[16];
	int16_t insamples[8][16];
	int16_t outsamples[8][14];
	int16_t last1 = 0, last2 = 0;
	int16_t coef1, coef2;
	int bestidx = 0;
	int scale[8];
	double distaccum[8];
	int v1, v2, v3;
	int distance, index;
	for (int framepos = 0; framepos < samplecount; framepos += 14){
		if (framepos % blocksamplecount == 0){
			seek[blockidx * 2] = last1;
			seek[blockidx * 2 + 1] = last2;
			blockidx += channelcount;
		}
		pcmbuf[0] = last2;
		pcmbuf[1] = last1;
		for (int j = 0; j < 14; j++){
			pcmbuf[j + 2] = pcm[framepos + j];
		}
		for (int coefidx = 0; coefidx < 8; coefidx++){
			coef1 = coefs[coefidx * 2];
			coef2 = coefs[coefidx * 2 + 1];
			insamples[coefidx][0] = last2;
			insamples[coefidx][1] = last1;
			distance = 0;
			for (int s = 0; s < 14; s++){
				insamples[coefidx][s + 2] = v1 = ((pcmbuf[s] * coef2) + (pcmbuf[s + 1] * coef1)) / 2048;
				v2 = pcmbuf[s + 2] - v1;
				v3 = CLAMP(v2, -32768, 32767);
				if (ABS(v3) > ABS(distance)){
					distance = v3;
				}
			}
			for (scale[coefidx] = 0; (scale[coefidx] <= 12) && ((distance > 7) || (distance < -8)); scale[coefidx]++, distance /= 2) {}
			scale[coefidx] = (scale[coefidx] <= 1) ? -1 : scale[coefidx] - 2;
			do {
				scale[coefidx]++;
				distaccum[coefidx] = 0;
				index = 0;
				for (int s = 0; s < 14; s++){
					v1 = ((insamples[coefidx][s] * coef2) + (insamples[coefidx][s + 1] * coef1));
					v2 = ((pcmbuf[s + 2] << 11) - v1) / 2048;
					v3 = (v2 > 0) ? (int)((double)v2 / (1 << scale[coefidx]) + 0.4999999f) : (int)((double)v2 / (1 << scale[coefidx]) - 0.4999999f);
					if (v3 < -8){
						if (index < (v3 = -8 - v3)){
							index = v3;
						}
						v3 = -8;
					} else if (v3 > 7){
						if (index < (v3 -= 7)){
							index = v3;
						}
						v3 = 7;
					}
					outsamples[coefidx][s] = v3;
					v1 = (v1 + ((v3 * (1 << scale[coefidx])) << 11) + 1024) >> 11;
					insamples[coefidx][s + 2] = v2 = CLAMP(v1, -32768, 32767);
					v3 = pcmbuf[s + 2] - v2;
					distaccum[coefidx] += v3 * (double)v3;
				}
				for (int x = index + 8; x > 256; x >>= 1){
					if (++scale[coefidx] >= 12){
						scale[coefidx] = 11;
					}
				}
			} while ((scale[coefidx] < 12) && (index > 1));
		}
		double mindist = DBL_MAX;
		for (int coefidx = 0; coefidx < 8; coefidx++){
			if (distaccum[coefidx] < mindist){
				mindist = distaccum[coefidx];
				bestidx = coefidx;
			}
		}
		if (framepos == 0){
			contexts[0] = bestidx;
			contexts[1] = scale[bestidx];
		} else if (loopstart >= framepos && loopstart < framepos + 14){
			contexts[2] = bestidx;
			contexts[3] = scale[bestidx];
		}
		*outpos++ = (bestidx << 4) | (scale[bestidx] & 0x0f);
		for (int j = 0; j < 7; j++){
			*outpos++ = (outsamples[bestidx][j * 2] << 4) | (outsamples[bestidx][j * 2 + 1] & 0x0f);
		}
		last1 = insamples[bestidx][15];
		last2 = insamples[bestidx][14];
	}
	return contexts;
}

PyObject* decodeDSPADPCMblock(PyObject* self, PyObject* args){
	PyArrayObject* adpcm_obj;
	PyArrayObject* pcmout_obj;
	PyArrayObject* coefs_obj;
	int samplecount;
	int blockstart;
	int last1;
	int last2;
	if (!PyArg_ParseTuple(args, "O!O!O!iiii", &PyArray_Type, &adpcm_obj, &PyArray_Type, &pcmout_obj, &PyArray_Type, &coefs_obj, &samplecount, &blockstart, &last1, &last2)){
		return NULL;
	}
	uint8_t* adpcm = (uint8_t*)PyArray_DATA(adpcm_obj);
	int16_t* pcmout = (int16_t*)PyArray_DATA(pcmout_obj);
	int16_t* coefs = (int16_t*)PyArray_DATA(coefs_obj);
	int* lasts = _decodeDSPADPCMblock(adpcm, pcmout, coefs, samplecount, blockstart, last1, last2);
	PyObject* result = Py_BuildValue("(ii)", lasts[0], lasts[1]);
	free(lasts);
	return result;
}

PyObject* encodeDSPADPCMchannel(PyObject* self, PyObject* args){
	PyArrayObject* coefs_obj;
	PyArrayObject* pcm_obj;
	PyArrayObject* adpcmout_obj;
	PyArrayObject* seek_obj;
	int samplecount;
	int blocksamplecount;
	int channelidx;
	int channelcount;
	int loopstart;
	if (!PyArg_ParseTuple(args, "O!O!O!O!iiiii", &PyArray_Type, &coefs_obj, &PyArray_Type, &pcm_obj, &PyArray_Type, &adpcmout_obj, &PyArray_Type, &seek_obj, &samplecount, &blocksamplecount, &channelidx, &channelcount, &loopstart)){
		return NULL;
	}
	int16_t* coefs = (int16_t*)PyArray_DATA(coefs_obj);
	int16_t* pcm = (int16_t*)PyArray_DATA(pcm_obj);
	uint8_t* adpcmout = (uint8_t*)PyArray_DATA(adpcmout_obj);
	int16_t* seek = (int16_t*)PyArray_DATA(seek_obj);
	int* contexts = _encodeDSPADPCMchannel(coefs, pcm, adpcmout, seek, samplecount, blocksamplecount, channelidx, channelcount, loopstart);
	PyObject* result = Py_BuildValue("iiii", contexts[0], contexts[1], contexts[2], contexts[3]);
	free(contexts);
	return result;
}

PyObject* generateDSPADPCMcoefs(PyObject* self, PyObject* args){
	PyArrayObject* coefs_obj;
	PyArrayObject* pcm_obj;
	int samplecount;
	if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &coefs_obj, &PyArray_Type, &pcm_obj, &samplecount)){
		return NULL;
	}
	int16_t* coefs = (int16_t*)PyArray_DATA(coefs_obj);
	int16_t* pcm = (int16_t*)PyArray_DATA(pcm_obj);
	_generateDSPADPCMcoefs(coefs, pcm, samplecount);
	return Py_BuildValue("i", 0);
}
