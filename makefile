CNN_VERSION = cnn_dist_v5
MAX_BUFFER = 4096
LV2_DIR = /usr/lib64/lv2/cnn_distortion
CXXFLAGS = -O3 -I/usr/local/include/lvtk-2 -Ilib/$(CNN_VERSION)/ -fPIC -std=c++17 -DCNN_VERSION=$(CNN_VERSION) -DMAX_BUFFER=$(MAX_BUFFER)
LDFLAGS = -lopenblas -L/usr/lib64

MODELS = $(wildcard models/*.pt)
LIBS = $(patsubst models/%.pt,lib/%/cnn_dist.h,$(MODELS)) $(patsubst models/%.pt,lib/%_half/cnn_dist.h,$(MODELS))

.SECONDARY: src/distortion.h

all: $(LIBS) bin/distortion.so

piode:
	g++ -O3 -I/usr/local/include/lvtk-2 -Ilib/$(CNN_VERSION)_half/ -fPIC -DCNN_VERSION=$(CNN_VERSION) -DMAX_BUFFER=256 -std=c++14 src/distortion.cpp -shared -o bin/distortion.so -lopenblas -L/usr/lib

piode-install:
	mkdir -p /usr/lib/lv2/cnn_distortion
	cp bin/distortion.so src/distortion.ttl src/manifest.ttl /usr/lib/lv2/cnn_distortion
	cp services/jackd.service services/cnn_distortion.service /etc/systemd/system/
	systemctl reload-daemon
	systemctl enable jackd
	systemctl enable cnn_distortion

install: bin/distortion.so
	mkdir -p $(LV2_DIR)
	cp bin/distortion.so src/distortion.ttl src/manifest.ttl $(LV2_DIR)

bin/%.so: src/%.cpp src/%.h src/%.ttl lib/$(CNN_VERSION)
	$(CXX) $(CXXFLAGS) $< -shared -o $@ $(LDFLAGS)

src/%.h: src/%.ttl
	lv2peg $< $@

lib/%/cnn_dist.h: models/%.pt
	@mkdir -p $$(dirname $@)
	python3 -m cnn_cpp $< -o $@

lib/%_half/cnn_dist.h: models/%.pt
	@mkdir -p $$(dirname $@)
	python3 -m cnn_cpp $< -o $@ -d 2
