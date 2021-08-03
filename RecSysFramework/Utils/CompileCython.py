#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/07/2017

@author: Anonymous Author
"""


try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension


from Cython.Distutils import build_ext


import numpy
import sys
import re



def compile_cython(fileToCompile):

    extensionName = re.sub("\.pyx", "", fileToCompile)

    ext_modules = Extension(extensionName,
                    [fileToCompile],
                    extra_compile_args=['-O3'],
                    include_dirs=[numpy.get_include(),],
                    )

    setup(
        cmdclass={'build_ext': build_ext},
        ext_modules=[ext_modules]
    )



if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError("Wrong number of paramethers received. Expected 2, got {}".format(sys.argv))

    # Get the name of the file to compile
    fileToCompile = sys.argv[1]
    # Remove the argument from sys argv in order for it to contain only what setup needs
    del sys.argv[1]

    compile_cython(fileToCompile)
