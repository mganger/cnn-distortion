import torch
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
		return str(m)
	else:
		return '{' + ','.join(mat_to_str(mi) for mi in m) + '}'

def sgemm(major,transa,transb,n,m,k,alpha,A,lda,B,ldb,beta,C,ldc):
	major = {'col':'CblasColMajor','row':'CblasRowMajor'}[major]

	d = {False:'CblasNoTrans',True:'CblasTrans'}
	transa,transb = d[transa],d[transb]

	
	return f"cblas_sgemm({major}, {transa}, {transb}, {n}, {m}, {k}, {alpha}, {A}, {lda}, {B}, {ldb}, {beta}, {C}, {ldc});"

nonlin = {
	nn.Tanh: ('Tanh (i.e. soft clip', 'std::tanhf(v)'),
	nn.Hardtanh: ('Hard Tanh (i.e. hard clip)','v > 1 ? 1 : v < -1 ? -1 : v'),
	nn.Softsign: ('Soft Sign x/(1+|x|)','v > 0 ? v/(1+v) : v/(1-v)'),
	LeakyTanh: ('Leaky Tanh','v > 1 ? 1+{l.a}f*(v-1) : v < -1 ? -1+{l.a}f*(v+1) : v'),
	Smoosh: ('Smoosher','v > {l.a}f ? v-{l.a}f : v < -{l.a}f ? v+{l.a}f : 0'),
	nn.ReLU: ('Rectified Linear Unit (ReLU)','v > 0 ? v : 0'),
	nn.LeakyReLU: ('Leaky Rectified Linear Unit (ReLU)','v > 0 ? v : {l.negative_slope}f*v'),
	SimpleTanh: ('Simple Tanh (polynomial)', f'v > 2 ? {4/3}f : v < -2 ? -{4/3}f : v - v*v*v*{1/12}f'),
	nn.Softshrink: ('Softshrink','v > {l.lambd}f ? v-{l.lambd}f : v < -{l.lambd}f ? v+{l.lambd}f : 0'),
	SoftSmoosh: ('Softsmoosh', 'v > 0 ? v*v/({l.l}f+v) : -v*v/({l.l}f-v)'),
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
	#latency = sum((l.kernel_size[0]-1)*(l.dilation[0]//divider) for l in seq if hasattr(l,'out_channels'))

	head = f"""#pragma once
#include <cnn_conv.h>

struct {classname} {{
"""

	func = f"""

	void operator()(float* x, float* y, int L) {{

		if (L > MAX_L || L <= latency) {{
			return;
		}}

		for (int i = 0; i < L; i++) {{
			x_odd[i][0] = x[i];
		}}

		//conv_nb(w, b, xi, xo, inch, ouch, ksize, inst, oust, dilation, length, maxch)
		//nonlin(xi,ch,length,st)
"""

	i = 0
	totallatency = 0
	xevenodd = ["x_even","x_odd"]
	xo = xevenodd[1]
	oust = 1
	for l in seq:
		if isinstance(l, (nn.Conv1d,nn.ConvTranspose1d)):
			inch, ouch = l.in_channels, l.out_channels
			ksize = l.kernel_size[0]
			dilation = l.dilation[0]
			inst = oust
			xi,xo = xo,xevenodd[i%2]
			s = l.stride[0]
			latency = inst*(ksize-1)*dilation
			totallatency += latency
			if isinstance(l,nn.Conv1d):
				oust = inst*s
			else:
				assert (inst%s) == 0, "Total stride must be an integer ratio"
				oust = inst//s

			if l.bias is None:
				l.bias = nn.Parameter(torch.zeros(ouch))
			if l is seq[-1]:
				l.bias = None
				l.bias = nn.Parameter(-seq(torch.zeros((1,1,totallatency+10)))[0,0,:1])

			name = f"layer_{i}"
			head += f"""
	static constexpr float w{i}[{ksize}][{inch}][{ouch}] = {mat_to_str(np.transpose(l.weight.detach().numpy().astype(np.float32),(2,1,0)))};
	static constexpr float b{i}[{ouch}] = {mat_to_str(l.bias.detach().numpy().astype(np.float32))};
"""

			func += f"""
		conv_nb(&w{i}, &b{i}, &{xi}, &{xo}, {inst}, {oust}, {dilation}, L-={latency});"""

			#latency += (l.kernel_size[0]-1)*(l.dilation[0]//divider)*stride*l.stride[0]
			#r += layer_to_str(l,f"layer_{i}",xi,xo,latency,stride,divider)
			#ch = l.out_channels
			#stride *= l.stride[0]
			i += 1
		elif isinstance(l, tuple(nonlin)):
			func += f"""
		relu(&{xo}, {ouch}, L, {oust});"""
		else:
			raise NotImplementedError(f"No converter for type {type(l)}")
	func += f"""

		// Copy result back to y
		for (int l = 0; l < L; l++) {{
			y[l+latency] = {xo}[l][0];
		}}
	}}
}};
"""
	max_ch = max(l.out_channels for l in seq if hasattr(l,'out_channels'))
	head += f"""

	const static int latency = {totallatency};
	const static int MAX_L = MAX_BUFFER + latency;
	const static int MAX_CH = {max_ch};
	// About {max_ch*2/1e6}*(MAX_BUFFER+{latency}) MB of buffer
	float x_even[MAX_L][MAX_CH];
	float x_odd [MAX_L][MAX_CH];

"""
	return head + func
