# Licensed under a 3-clause BSD style license - see LICENSE.rst
from setuptools import setup

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

setup(name='Ska.Sun',
      author='Jean Connelly',
      description='Sun position calculator',
      author_email='jconnelly@cfa.harvard.edu',
      use_scm_version=True,
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      zip_safe=False,
      packages=['Ska', 'Ska.Sun'],
      package_data={},
      tests_require=['pytest'],
      cmdclass=cmdclass,
      )
