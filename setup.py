#!/usr/bin/env python
from setuptools import setup
import sys

long_description = ''

if 'upload' in sys.argv:
    with open('README.rst') as f:
        long_description = f.read()

install_reqs = [
    'funcsigs>=0.4',
    'matplotlib>=1.4.0',
    'mock>=1.1.2',
    'numpy>=1.9.1',
    'pandas>=0.16.1',
    'pyparsing>=2.0.3',
    'python-dateutil>=2.4.2',
    'pytz>=2014.10',
    'scipy>=0.14.0',
    'seaborn>=0.6.0',
    'statsmodels>=0.6.1'
]

test_reqs = [
    "nose>=1.3.7",
    "nose_parameterized>=0.5.0"
]

if __name__ == "__main__":
    setup(
        name='qfactor',
        version='0.0.0',
        description='Factor analysis tools',
        author='Quantopian Inc.',
        author_email='andrew@quantopian.com',
        packages=[
            'qfactor',
        ],
        long_description=long_description,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Natural Language :: English',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python',
            'Topic :: Utilities',
        ],
        url='https://github.com/quantopian/qfactor',
        install_requires=install_reqs,
        test_requires=test_reqs
    )