CNN Distortion
==============

This is a project that demonstrates it's actually possible to combine deep
learning and DSP (digital signal processing). It's a product of a recent
epiphany that I had: CNNs (convolutional neural networks) are just the
non-linear extension of FIR (finite impulse response) filters.

The Goal
--------

Quite simply, I wanted to replicate the distortion of my tube guitar amp on my
laptop. How, you ask? Using a CNN! Here's the process:

1. Record a clean guitar signal.
2. Play back the clean guitar signal through the distortion (amp/pedal/anything).
3. Train a CNN to produce the distorted signal from the clean signal.
4. Implement the CNN in a realtime plugin.

The Result
----------

Basically, the result is that it's super cool! Checkout the audio clips in my
blog post [here](https://michaelganger.org/articles/index.php/2019/10/31/from-numbers-to-rockstar/).

<audio controls="" src="https://michaelganger.org/articles/wp-content/uploads/2019/10/medium-1.mp3"></audio>

To any audiophiles that think it sounds weird, the reason is simple: I recorded
the output straight from the headphone output of the amp (my Orange Micro
Terror), not through a speaker cabinet/microphone. The model likewise
reproduces the sound of the amp, not the combined sound of the amp, speaker,
microphone, and room.

An added benefit of this approach is that it adds a negligible amount of noise
to the signal (basically only from roundoff errors from adding/multiplying 32
bit floats). No more hiss from the amp (the guitar is still pretty noisy)!

Doing It Yourself
-----------------

Let's be real, you came to this page because you wanted to 1) use the plugin or
2) do the same thing with your own amp. Before you do so, I encourage you to read
through the blog posts I wrote about it to get a feel for what the process
looks like. Here's the main tools/frameworks I used:

 - PyTorch
 - LV2
 - BLAS

The general process is as follows; I assume that you bring some level of Deep
Learning and DSP knowledge:

1. Record the training data
2. Play around with CNN architectures in PyTorch until you get one that seems to work.
3. Save the model and convert it to a C++ header file.
4. Use the model in your plugin by `#include`ing it (or just modifying my example plugin).

You can install the PyTorch to C++ converter by installing the included python library:

```bash
cd python
pip3 install --user .    # If you don't want to make any changes to the code
pip3 install -e --user . # If you want to change the code
```

To build and install the LV2 plugin, run:

```bash
make
sudo make install
```

The build requirements are:
 - LV2 [https://github.com/drobilla/lv2](https://github.com/drobilla/lv2)
   (you should use your package manager for this if possible)
 - LV2 Toolkit [https://github.com/lvtk/lvtk](https://github.com/lvtk/lvtk)
 - OpenBLAS

At the moment, the makefile is pretty basic. You may need to modify `LV2_DIR`
to suit your needs.  If you build a new model, put it in `models/` with a `.pt`
extension and add the header directly to the `all` target in the makefile:

```makefile
all: lib/cnn_dist_v1.h lib/cnn_dist_v2.h lib/your_awesome_model.h bin/distortion.so
```

To use your fancy model instead of the default one, you can modify
`distortion.cpp` to accomodate:

```c++
...
#include <your_awesome_model.h>
using cnn_dist = your_awesome_model_class;
...
```

It *should* work fine. If it doesn't, an issue or pull request would be much
appreciated.
