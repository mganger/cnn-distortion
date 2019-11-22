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

def layer_to_str(layer, name, xi, xo, latency, divider=1):
	m = layer.in_channels
	n = layer.out_channels
	d, = layer.dilation
	k, = layer.kernel_size
	d //= divider

	weights = mat_to_str(np.moveaxis(layer.weight.detach().numpy(),2,0))
	if layer.bias is None:
		r = f"""
		// auto-generated code for layer {name}: {layer}
		const float w_{name}[{k}][{n}][{m}] = {weights};

		// Apply main filter for {name}
		// {xo}[:,{latency}:] = sum(w[k]@{xi}[:,{latency}-({k-1}-k)*{d}:L-({k-1}-k)*{d}] for k in w.shape[0])
		for (int k = 0; k < {k}; k++) {{
			int offset = ({k-1}-k)*{d};
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, {n}, L-{latency}, {m}, 1.0, &w_{name}[k][0][0], {m}, &{xi}[0][{latency}-offset], MAX_L, k==0?0.0:1.0, &{xo}[0][{latency}], MAX_L);
		}}

"""
	else:
		bias = mat_to_str(layer.bias)

		r = f"""
		// auto-generated code for layer {name}: {layer}
		const float w_{name}[{k}][{n}][{m}] = {weights};
		const float b_{name}[{n}] = {bias};

		// Fill with biases for {name}
		for (int i = 0; i < {n}; i++) {{
			for (int l = {latency}; l < L; l++) {{
				{xo}[i][l] = b_{name}[i];
			}}
		}}

		// Apply main filter for {name}
		// {xo}[:,{latency}:] = sum(w[k]@{xi}[:,{latency}-({k-1}-k)*{d}:L-({k-1}-k)*{d}] for k in w.shape[0])
		for (int k = 0; k < {k}; k++) {{
			int offset = ({k-1}-k)*{d};
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, {n}, L-{latency}, {m}, 1.0, &w_{name}[k][0][0], {m}, &{xi}[0][{latency}-offset], MAX_L, 1.0, &{xo}[0][{latency}], MAX_L);
		}}

"""
	return r

def tanh_to_str(size,xi,xo,latency):
	r = f"""
		// Tanh (i.e. soft clip)
		for (int i = 0; i < {size}; i++) {{
			for (int l = {latency}; l < L; l++) {{
				{xo}[i][l] = std::tanh({xi}[i][l]);
			}}
		}}

"""
	return r

def hardtanh_to_str(size,xi,xo,latency):
	r = f"""
		// Hard Tanh (i.e. hard clip)
		for (int i = 0; i < {size}; i++) {{
			for (int l = {latency}; l < L; l++) {{
				auto v = {xi}[i][l];
				{xo}[i][l] = v > 1 ? 1 : v < -1 ? -1 : v;
			}}
		}}

"""
	return r

def softsign_to_str(size,xi,xo,latency):
	r = f"""
		// Soft Sign x/(1+|x|)
		for (int i = 0; i < {size}; i++) {{
			for (int l = {latency}; l < L; l++) {{
				auto v = {xi}[i][l];
				{xo}[i][l] = v > 0 ? v/(1+v) : v/(1-v);
			}}
		}}

"""
	return r

def leakytanh_to_str(a,size,xi,xo,latency):
	r = f"""
		// Leaky Tanh
		for (int i = 0; i < {size}; i++) {{
			for (int l = {latency}; l < L; l++) {{
				auto v = {xi}[i][l];
				{xo}[i][l] = v > 1 ? 1+{a}*(v-1) : v < -1 ? -1+{a}*(v+1) : v;
			}}
		}}

"""
	return r

def smoosh_to_str(a,size,xi,xo,latency):
	r = f"""
		// Smoosher
		for (int i = 0; i < {size}; i++) {{
			for (int l = {latency}; l < L; l++) {{
				auto v = {xi}[i][l];
				{xo}[i][l] = v > {a}f ? v-{a}f : v < -{a}f ? v+{a}f : 0;
			}}
		}}

"""
	return r

def relu_to_str(size,xi,xo,latency):
	r = f"""
		// Rectified Linear Unit (ReLU)
		for (int i = 0; i < {size}; i++) {{
			for (int l = {latency}; l < L; l++) {{
				{xo}[i][l] = {xi}[i][l] > 0 ? {xi}[i][l] : 0;
			}}
		}}

"""
	return r

def leaky_relu_to_str(a,size,xi,xo,latency):
	r = f"""
		// Leaky Rectified Linear Unit (ReLU)
		for (int i = 0; i < {size}; i++) {{
			for (int l = {latency}; l < L; l++) {{
				{xo}[i][l] = {xi}[i][l] > 0 ? {xi}[i][l] : {a}f*{xi}[i][l];
			}}
		}}

"""
	return r

def simpletanh_to_str(size,xi,xo,latency):
	r = f"""
		// Leaky Rectified Linear Unit (ReLU)
		for (int i = 0; i < {size}; i++) {{
			for (int l = {latency}; l < L; l++) {{
				auto v = {xi}[i][l];
				{xo}[i][l] = v > 2 ? {4/3} : v < -2 ? -{4/3} : v - v*v*v*{1/12};
			}}
		}}

"""
	return r

def softshrink_to_str(lambd,size,xi,xo,latency):
	r = f"""
		// Softshrink
		for (int i = 0; i < {size}; i++) {{
			for (int l = {latency}; l < L; l++) {{
				auto v = {xi}[i][l];
				{xo}[i][l] = v > {lambd}f ? v-{lambd}f : v < -{lambd}f ? v+{lambd}f : 0;
			}}
		}}
"""
	return r

def softsmoosh_to_str(lambd,size,xi,xo,latency):
	r = f"""
		// Softshrink
		for (int i = 0; i < {size}; i++) {{
			for (int l = {latency}; l < L; l++) {{
				auto v = {xi}[i][l];
				{xo}[i][l] = v > 0 ? v*v/({lambd}f+v) : -v*v/({lambd}f-v);
			}}
		}}
"""
	return r

def sequential_to_str(seq, classname, divider=1):
	max_w = max(l.out_channels for l in seq if hasattr(l,'out_channels'))
	latency = sum((l.kernel_size[0]-1)*(l.dilation[0]//divider) for l in seq if hasattr(l,'out_channels'))

	r = f"""
extern "C" {{
#include <cblas.h>
}};

#include <cmath>

struct {classname} {{
	const static int latency = {latency};
	const static int MAX_L = MAX_BUFFER + latency;
	// About {max_w*2/1e6}*(MAX_BUFFER+{latency}) MB of buffer
	float x_even[{max_w}][MAX_L];
	float x_odd [{max_w}][MAX_L];

	void operator()(float* x, float* y, int L) {{

		// Ensure we don't segfault
		L = L > MAX_L ? MAX_L : L;

		for (int i = 0; i < L; i++) {{
			x_odd[0][i] = x[i];
		}}
"""

	s = None
	i = 0
	latency = 0
	xevenodd = ["x_even","x_odd"]
	xi = xevenodd[1]
	for l in seq:
		if isinstance(l, nn.ReLU):
			r += relu_to_str(s,xi,xi,latency)
		elif isinstance(l, nn.LeakyReLU):
			r += leaky_relu_to_str(l.negative_slope, s,xi,xi,latency)
		elif isinstance(l, Smoosh):
			r += smoosh_to_str(l.gap,s,xi,xi,latency)
		elif isinstance(l, nn.Hardtanh):
			r += hardtanh_to_str(s,xi,xi,latency)
		elif isinstance(l, nn.Tanh):
			r += tanh_to_str(s,xi,xi,latency)
		elif isinstance(l, LeakyTanh):
			r += leakytanh_to_str(l.a,s,xi,xi,latency)
		elif isinstance(l, SimpleTanh):
			r += simpletanh_to_str(s,xi,xi,latency)
		elif isinstance(l, nn.Softsign):
			r += softsign_to_str(s,xi,xi,latency)
		elif isinstance(l, nn.Softshrink):
			r += softshrink_to_str(l.lambd,s,xi,xi,latency)
		elif isinstance(l, SoftSmoosh):
			r += softsmoosh_to_str(l.l,s,xi,xi,latency)
		else:
			latency += (l.kernel_size[0]-1)*(l.dilation[0]//divider)
			#xo = f"x{i}"
			xo = xevenodd[i%2]
			r += layer_to_str(l,f"layer_{i}",xi,xo,latency,divider)
			s = l.out_channels
			i += 1
			xi = xo
	r += f"""
		// Copy result back to y
		for (int l = {latency}; l < L; l++) {{
			y[l] = {xo}[0][l];
		}}
	}}
}};
"""
	return r
