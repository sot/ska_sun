# Licensed under a 3-clause BSD style license - see LICENSE.rst
from setuptools import setup

from Ska.Sun import __version__

setup(name='Ska.Sun',
      author = 'Jean Connelly',
      description='Sun position calculator',
      author_email = 'jeanconn@head.cfa.harvard.edu',
      py_modules = ['Ska.Sun'],
      version=__version__,
      zip_safe=False,
      packages=['Ska'],
      package_dir={'Ska' : 'Ska'},
      package_data={}
      )
