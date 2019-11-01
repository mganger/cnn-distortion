from distutils.core import setup

setup(
	name="cnn_cpp",
	version="0.1",
	packages=["cnn_cpp"],
	license='MIT',
	author="Michael Ganger",
	author_email="mg@michaelganger.org",
	install_requires=[
		'torch',
		'numpy',
	],
)
