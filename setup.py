# Licensed under a 3-clause BSD style license - see LICENSE.rst
from setuptools import setup

setup(name='Ska.Sun',
      author = 'Jean Connelly',
      description='Sun position calculator',
      author_email = 'jeanconn@head.cfa.harvard.edu',
      py_modules = ['Ska.Sun'],
      use_scm_version=True,
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      zip_safe=False,
      packages=['Ska'],
      package_dir={'Ska' : 'Ska'},
      package_data={}
      )
