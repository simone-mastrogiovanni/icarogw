import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='icarogw',
    version='2.0.2',
    author='Simone Mastrogiovanni',
    author_email='simone.mastrogiovanni@ligo.org',
    description='A python package for inference of population properties of noisy, heterogeneous and incomplete observations',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/simone-mastrogiovanni/icarogw',
    license='EUPL-1.2',
    python_requires='>=3.10',
    packages=['icarogw'],
    install_requires=['attrs==23.1.0','tables==3.9.1','jupyterlab==4.0.8','bilby==2.2.0',
                      'healpy==1.16.6','mpmath==1.3.0','ligo.skymap==1.1.2','ChainConsumer==1.0.2']
)
