import torch.nn as nn
import numpy as np

class Smoosh(nn.Module):
    def __init__(self,gap=0.1):
        super().__init__()
        self.gap = gap
    def forward(self, x):
        return F.relu(x-self.gap)-F.relu(-self.gap-x)

    
class LeakyTanh(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.a = a
    def forward(self, x):
        return x*self.a + F.hardtanh(x)*(1-self.a)

class SimpleTanh(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.empty_like(x)
        r1 = x>2
        r2 = x<-2
        r3 = (~r1)&(~r2)
        y[r1] = 4/3
        y[r2] = -4/3
        y[r3] = x[r3]-x[r3]**3/12
        return y

class SoftSmoosh(nn.Module):
    def __init__(self,l):
        super().__init__()
        self.l = l
        self.act = nn.Softsign()
    def forward(self, x):
        return x-self.act(x/self.l)*self.l


def mat_to_str(m):
	if len(m.shape) == 0:
		return str(m.item())
	else:
		return '{' + ','.join(mat_to_str(mi) for mi in m) + '}'

def sgemm(major,transa,transb,n,m,k,alpha,A,lda,B,ldb,beta,C,ldc):
	major = {'col':'CblasColMajor','row':'CblasRowMajor'}[major]

	d = {False:'CblasNoTrans',True:'CblasTrans'}
	transa,transb = d[transa],d[transb]

	
	return f"cblas_sgemm({major}, {transa}, {transb}, {n}, {m}, {k}, {alpha}, {A}, {lda}, {B}, {ldb}, {beta}, {C}, {ldc});"


def layer_to_str(layer, name, xi, xo, latency, divider=1):
	m = layer.in_channels
	n = layer.out_channels
	d, = layer.dilation
	k, = layer.kernel_size
	d //= divider

	weights = mat_to_str(np.transpose(layer.weight.detach().numpy(),(2,1,0)))
	if layer.bias is None:
		return f"""
		// auto-generated code for layer {name}: {layer}
		const float w_{name}[{k}][{m}][{n}] = {weights};

		// Apply main filter for {name}
		// {xo}[:,{latency}:] = sum(w[k]@{xi}[:,{latency}-({k-1}-k)*{d}:L-({k-1}-k)*{d}] for k in w.shape[0])
		for (int k = 0; k < {k}; k++) {{
			int offset = ({k-1}-k)*{d};
			float beta = k == 0 ? 0.0 : 1.0;
			""" + sgemm('col',False,False,n,f"L-{latency}",m,1.0,f"&w_{name}[k][0][0]",n,f"&{xi}[{latency}-offset][0]","MAX_CH","beta",f"&{xo}[{latency}][0]","MAX_CH") + """
		}

"""
	else:
		bias = mat_to_str(layer.bias)

		return f"""
		// auto-generated code for layer {name}: {layer}
		const float w_{name}[{k}][{m}][{n}] = {weights};
		const float b_{name}[{n}] = {bias};

		// Fill with biases for {name}
		for (int l = {latency}; l < L; l++) {{
			for (int i = 0; i < {n}; i++) {{
				{xo}[l][i] = b_{name}[i];
			}}
		}}

		// Apply main filter for {name}
		// {xo}[:,{latency}:] = sum(w[k]@{xi}[:,{latency}-({k-1}-k)*{d}:L-({k-1}-k)*{d}] for k in w.shape[0])
		for (int k = 0; k < {k}; k++) {{
			int offset = ({k-1}-k)*{d};
			""" + sgemm('col',False,False,n,f"L-{latency}",m,1.0,f"&w_{name}[k][0][0]",n,f"&{xi}[{latency}-offset][0]","MAX_CH",1.0,f"&{xo}[{latency}][0]","MAX_CH") + """
		}

"""

nonlin = {
	nn.Tanh: ('Tanh (i.e. soft clip', 'std::tanhf(v)'),
	nn.Hardtanh: ('Hard Tanh (i.e. hard clip)','v > 1 ? 1 : v < -1 ? -1 : v'),
	nn.Softsign: ('Soft Sign x/(1+|x|)','v > 0 ? v/(1+v) : v/(1-v)'),
	nn.Tanh: ('Leaky Tanh','v > 1 ? 1+{l.a}f*(v-1) : v < -1 ? -1+{l.a}f*(v+1) : v'),
	Smoosh: ('Smoosher','v > {l.a}f ? v-{l.a}f : v < -{l.a}f ? v+{l.a}f : 0'),
	nn.ReLU: ('Rectified Linear Unit (ReLU)','v > 0 ? v : 0'),
	nn.LeakyReLU: ('Leaky Rectified Linear Unit (ReLU)','v > 0 ? v : {a}f*v'),
	SimpleTanh: ('Simple Tanh (polynomial)', f'v > 2 ? {4/3}f : v < -2 ? -{4/3}f : v - v*v*v*{1/12}f'),
	nn.Softshrink: ('Softshrink','v > {l.lambd}f ? v-{l.lambd}f : v < -{l.lambd}f ? v+{l.lambd}f : 0'),
	SoftSmoosh: ('Softsmoosh', 'v > 0 ? v*v/({l.lambd}f+v) : -v*v/({l.lambd}f-v)'),
}

def nonlin_to_str(layer,chan,xi,xo,latency):
	comment, code = nonlin[type(layer)]
	code = code.format(l=layer)
	return f"""
		// {comment}
		for (int l = {latency}; l < L; l++) {{
			for (int i = 0; i < {chan}; i++) {{
				auto& v = {xi}[i][l];
				{xo}[l][i] = {code};
			}}
		}}

"""

def sequential_to_str(seq, classname, divider=1):
	max_w = max(l.out_channels for l in seq if hasattr(l,'out_channels'))
	latency = sum((l.kernel_size[0]-1)*(l.dilation[0]//divider) for l in seq if hasattr(l,'out_channels'))

	r = f"""
#pragma once
extern "C" {{
#include <cblas.h>
}};

#include <cmath>

struct {classname} {{
	const static int latency = {latency};
	const static int MAX_L = MAX_BUFFER + latency;
	const static int MAX_CH = {max_w};
	// About {max_w*2/1e6}*(MAX_BUFFER+{latency}) MB of buffer
	float x_even[MAX_L][MAX_CH];
	float x_odd [MAX_L][MAX_CH];

	void operator()(float* x, float* y, int L) {{

		// Ensure we don't segfault
		L = L > MAX_L ? MAX_L : L;

		for (int i = 0; i < L; i++) {{
			x_odd[i][0] = x[i];
		}}
"""

	s = None
	i = 0
	latency = 0
	xevenodd = ["x_even","x_odd"]
	xi = xevenodd[1]
	for l in seq:
		if isinstance(l, tuple(nonlin)):
			r += nonlin_to_str(l,s,xi,xi,latency)
		elif isinstance(l, nn.Conv1d):
			latency += (l.kernel_size[0]-1)*(l.dilation[0]//divider)
			#xo = f"x{i}"
			xo = xevenodd[i%2]
			r += layer_to_str(l,f"layer_{i}",xi,xo,latency,divider)
			s = l.out_channels
			i += 1
			xi = xo
		else:
			raise NotImplementedError(f"No converter for type {type(l)}")
	r += f"""
		// Copy result back to y
		for (int l = {latency}; l < L; l++) {{
			y[l] = {xo}[l][0];
		}}
	}}
}};
"""
	return r
