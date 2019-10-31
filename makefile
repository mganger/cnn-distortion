LV2_DIR = /usr/lib64/lv2/cnn_distortion
CXXFLAGS = -O3 -I/usr/local/include/lvtk-2 -Ilib/ -fPIC -std=c++17
LDFLAGS = -lopenblas -L/usr/lib64

.SECONDARY: src/distortion.h

all: lib/cnn_dist_v1.h lib/cnn_dist_v2.h bin/distortion.so

install: bin/distortion.so lib/cnn_dist_v1.h
	mkdir -p $(LV2_DIR)
	cp bin/distortion.so src/distortion.ttl src/manifest.ttl $(LV2_DIR)

bin/%.so: src/%.cpp src/%.h src/%.ttl
	$(CXX) $(CXXFLAGS) $< -shared -o $@ $(LDFLAGS)

src/%.h: src/%.ttl
	lv2peg $< $@

lib/%.h: models/%.pt
	python3 -m cnn_cpp $< -o $@
