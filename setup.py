from setuptools import setup
setup(name='Ska.Sun',
      author = 'Jean Connelly',
      description='Sun position calculator',
      author_email = 'jeanconn@head.cfa.harvard.edu',
      py_modules = ['Ska.Sun'],
      version='0.02',
      zip_safe=False,
      namespace_packages=['Ska'],
      packages=['Ska'],
      package_dir={'Ska' : 'Ska'},
      package_data={}
      )
