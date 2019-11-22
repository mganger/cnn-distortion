#pragma once

extern "C" {
#include <cblas.h>
};

#include <cmath>

template <int inch, int ouch, int ksize, int maxch, int maxl>
inline void conv_nb(const float (*w)[ksize][inch][ouch], const float (*b)[ouch], float (*xi)[maxl][maxch], float (*xo)[maxl][maxch], int inst, int oust, int dilation, int length) {

	const int minst = std::min(inst,oust);
	const int maxst = std::max(inst,oust);
	const int lengthst = length/maxst;
	const int lda = ouch;
	const int ldb = maxst*maxch;
	const int ldc = ldb;
	const int koffset = dilation*inst;
	const int reach = ksize*koffset;

	for (int l = 0; l < length; l += oust) {
		for (int i = 0; i < ouch; i++) {
			(*xo)[l][i] = (*b)[i];
		}
	}

	for (int o = 0; o < maxst; o += oust) {
		for (int i = o; i < reach+o; i += koffset) {
			if (((i-o) % inst) == 0) {
				const float* wk = &(*w)[(i-o)/koffset][0][0];
				float* in = &(*xi)[i][0];
				float* out = &(*xo)[o][0];
				cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ouch, lengthst, inch, 1.0, wk, lda, in, ldb, 1.0, out, ldc);
			}
		}
	}
}

template <int maxl,int maxch>
inline void relu(float (*x)[maxl][maxch], int ch, int length, int offset) {
	for (int i = 0; i < length; i+=offset) {
		for (int j = 0; j < ch; j++) {
			(*x)[i][j] = (*x)[i][j] < 0 ? 0 : (*x)[i][j];
		}
	}
}
