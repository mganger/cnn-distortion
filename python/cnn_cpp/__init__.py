import torch.nn as nn
import numpy as np

def mat_to_str(m):
    if len(m.shape) == 0:
        return str(m.item())
    else:
        return '{' + ','.join(mat_to_str(mi) for mi in m) + '}'
    
def layer_to_str(layer, name, xi, xo, latency):
    m = layer.in_channels
    n = layer.out_channels
    d, = layer.dilation
    k, = layer.kernel_size

    weights = mat_to_str(np.moveaxis(layer.weight.detach().numpy(),2,0))
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

def sequential_to_str(seq, classname):
    
    max_w = max(l.out_channels for l in seq if hasattr(l,'out_channels'))
    latency = sum((l.kernel_size[0]-1)*l.dilation[0] for l in seq if hasattr(l,'out_channels'))
    
    r = f"""
extern "C" {{
#include <cblas.h>
}};

struct {classname} {{
    const static int latency = {latency};
    const static int MAX_L = {8192+latency};
    // About {max_w*2*(8192+latency)/1e6} MB of buffer
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
        else:
            latency += (l.kernel_size[0]-1)*l.dilation[0]
            #xo = f"x{i}"
            xo = xevenodd[i%2]
            r += layer_to_str(l,f"layer_{i}",xi,xo,latency)
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
