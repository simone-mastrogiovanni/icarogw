import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='icarogw',
    version='2.0.0',
    author='Simone Mastrogiovanni',
    author_email='simone.mastrogiovanni@ligo.org',
    description='A python package for inference of population properties of noisy, heterogeneous and incomplete observations',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/simone-mastrogiovanni/icarogw',
    license='GNU GPLv3',
    python_requires='>=3.8',
    packages=['icarogw'],
    install_requires=['bilby>=2.1.0','healpy>=1.16.2','mpmath>=1.3.0','ligo.skymap>=1.0.7']
)
