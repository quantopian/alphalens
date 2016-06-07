#!/usr/bin/env python
from setuptools import setup
import sys

long_description = ''

if 'upload' in sys.argv:
    with open('README.md') as f:
        long_description = f.read()
if __name__ == "__main__":
    setup(
        name='pyfactor',
        version='0.0.0',
        description='Factor analysis tools',
        author='Quantopian Inc.',
        author_email='andrew@quantopian.com',
        packages=[
            'pyfactor',
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
        url='https://github.com/quantopian/pyfactor',
        install_requires=()
    )