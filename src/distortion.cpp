#include <lvtk/plugin.hpp>
#include <boost/circular_buffer.hpp>
#include "distortion.h"
#include <cnn_dist_v1.h>
#include <cmath>

float from_db(float x) {
	return std::pow(10,x/20);
}

using cnn_dist = cnn_dist_v1;

class CnnDistortion : public lvtk::Plugin<CnnDistortion> {
private:
	lvtk::Args args;
	boost::circular_buffer<float> b;
	cnn_dist algo;

	float* port[p_n_ports];

public:
	CnnDistortion(const lvtk::Args& args_) :
		Plugin(args_), args(args_), b(cnn_dist::MAX_L, 0.0f)
	{}

	void connect_port (uint32_t p, void* data) {
		port[p] = static_cast<float*>(data);
	}

	void run (uint32_t nframes) {
		b.insert(b.end(), port[p_in], port[p_in]+nframes);
		int L = nframes+algo.latency;
		float xi[L], xo[L];
		float pregain = from_db(*port[p_gain]);
		float postgain = from_db(*port[p_makeup] - *port[p_gain]);
		std::transform(b.end()-L-1,b.end(),xi,[pregain](float x){return x*pregain;});
		algo(xi,xo,L);
		std::transform(xo+L-nframes,xo+L,port[p_out],[postgain](float x){return x*postgain;});
	}
};

static const lvtk::Descriptor<CnnDistortion> cnn_dist_desc(p_uri);
