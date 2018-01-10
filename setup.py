#!/usr/bin/env python
from setuptools import setup, find_packages
import versioneer
import sys

long_description = ''

if 'upload' in sys.argv:
    with open('README.rst') as f:
        long_description = f.read()

install_reqs = [
    'matplotlib>=1.4.0',
    'numpy>=1.9.1',
    'pandas>=0.18.0',
    'scipy>=0.14.0',
    'seaborn>=0.6.0',
    'statsmodels>=0.6.1',
    'IPython>=3.2.3',
]

extra_reqs = {
    'test': [
        "nose>=1.3.7",
        "parameterized>=0.5.0",
        "tox>=2.3.1",
    ],
}

if __name__ == "__main__":
    setup(
        name='alphalens',
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        description='Performance analysis of predictive (alpha) stock factors',
        author='Quantopian Inc.',
        author_email='opensource@quantopian.com',
        packages=find_packages(include='alphalens.*'),
        package_data={
            'alphalens': ['examples/*'],
        },
        long_description=long_description,
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Natural Language :: English',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python',
            'Topic :: Utilities',
            'Topic :: Office/Business :: Financial',
            'Topic :: Scientific/Engineering :: Information Analysis',
        ],
        url='https://github.com/quantopian/alphalens',
        install_requires=install_reqs,
        extras_require=extra_reqs,
    )
