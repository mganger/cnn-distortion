#ifndef src_distortion_h
#define src_distortion_h


#ifndef PEG_STRUCT
#define PEG_STRUCT
typedef struct {
  float min;
  float max;
  float default_value;
  char toggled;
  char integer;
  char logarithmic;
} peg_data_t;
#endif

/* <https://michaelganger.org/plugins/cnn_distortion> */

static const char p_uri[] = "https://michaelganger.org/plugins/cnn_distortion";

enum p_port_enum {
  p_in,
  p_out,
  p_gain,
  p_makeup,
  p_n_ports
};

static const peg_data_t p_ports[] = {
  { -3.40282e+38, 3.40282e+38, -3.40282e+38, 0, 0, 0 }, 
  { -3.40282e+38, 3.40282e+38, -3.40282e+38, 0, 0, 0 }, 
  { -90, 24, 0, 0, 0, 0 }, 
  { -90, 24, 0, 0, 0, 0 }, 
};


#endif /* src_distortion_h */
