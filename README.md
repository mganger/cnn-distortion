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

The Process
-----------

I used PyTorch in a Jupyter notebook to train the CNN. It's in the repository,
but you can see the result
[here](https://nbviewer.jupyter.org/github/mganger/cnn-distortion/blob/master/notebooks/CNN%20Distortion.ipynb).

The Result
----------

Basically, the result is that it's super cool! Checkout the audio clips in my
blog post [here](https://michaelganger.org/articles/index.php/2019/10/31/from-numbers-to-rockstar/).

To any audiophiles that think it sounds weird, the reason is simple: I recorded
the output straight from the headphone output of the amp (my Orange Micro
Terror), not through a speaker cabinet/microphone. The model likewise
reproduces the sound of the amp, not the combined sound of the amp, speaker,
microphone, and room.

An added benefit of this approach is that it adds a negligible amount of noise
to the signal (basically only from roundoff errors from adding/multiplying 32
bit floats). No more hiss from the amp (the guitar is still pretty noisy)!

Running on a Raspberry PI
-------------------------

If you're interested in modifying the model, learning, or otherwise doing any
customization, see the next section on Doing It Yourself. If you're running it
on a Raspberry Pi, you're in luck! It's not too hard (with a couple
modifications that I made).

### Requirements:

 - Raspberry Pi v2+ (with some access to the commandline).
 - USB Audio Interface
 - SD card with Raspbian Buster (or newer)

### Setup:

1. Clone `git@github.com:mganger/cnn-distortion`.
2. Clone `https://github.com/lvtk/lvtk` and install using the instructions there.
3. Run `sudo apt install libopenblas-dev libboost-dev lv2-dev jalv lv2-c++-tools`.
4. Build with `make piode`.
5. Determine the Alsa device you would like to use with `cat /proc/asound/cards` (see [here](https://jackaudio.org/faq/device_naming.html)).
6. Update the `jackd.service` systemd service to use the right sound card.
   Optionally, adjust the other parameters to your liking (but if you make the
   period size larger than 256, make sure you update the `-DMAX_BUFFER=...`
   argument under the `piode:` target).
7. Install with `sudo make piode `.
8. Reboot, plug in the audio device, and enjoy!

For a video of this in action, see [this post](https://michaelganger.org/articles/index.php/2019/11/05/piode/).

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
3. Save the model to the `models/` folder, and update the `CNN_VERSION`
   variable in the makefile accordingly.

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
 - LV2 - [https://github.com/drobilla/lv2](https://github.com/drobilla/lv2)
   (you should use your package manager for this if possible)
 - LV2 Toolkit - [https://github.com/lvtk/lvtk](https://github.com/lvtk/lvtk)
 - OpenBLAS
 - Boost

At the moment, the makefile is pretty basic. You may need to modify `LV2_DIR`
to suit your needs.  If you build a new model, put it in `models/` with a `.pt`
extension.  To use your fancy model instead of the default one, just modify the
`CNN_VERSION` variable in the makefile to accomodate (drop the `.pt` extension):

```makefile
...
CNN_VERSION = your_awesome_model
...
```

It *should* work fine. If it doesn't, an issue or pull request would be much
appreciated.
