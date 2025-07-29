import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='icarogw',
    version='2.0.3',
    author='Simone Mastrogiovanni, Gregoire Pierra, Vasco Gennari, Sarah Ferraiuolo, Leonardo Iampieri',
    author_email='simone.mastrogiovanni@ligo.org',
    description='A python package for inference of population properties of noisy, heterogeneous and incomplete observations',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/simone-mastrogiovanni/icarogw',
    license='EUPL-1.2',
    python_requires='>=3.12',
    packages=['icarogw'],
    install_requires=['bilby==2.6.0','mhealpy==0.3.6',
                     'ligo.skymap==2.4.0','mpmath==1.3.0','seaborn==0.13.2','nessai-bilby==0.1.0.post0']
)
