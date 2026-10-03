import sys
from setuptools import setup,Extension
setup(name='tracer-runtime',version='0.1.0',packages=['tracer_runtime'],python_requires='>=3.12',ext_modules=[Extension('tracer_runtime._legacy_math',['tracer_runtime/_legacy_math.c'],extra_compile_args=[] if sys.platform=='win32' else ['-O2','-fno-fast-math','-ffp-contract=off'])])
