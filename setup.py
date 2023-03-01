# Licensed under a 3-clause BSD style license - see LICENSE.rst

from setuptools import setup
from ska_helpers.setup_helper import duplicate_package_info
from testr.setup_helper import cmdclass

name = "ska_sun"
namespace = "Ska.Sun"

packages = ["ska_sun", "ska_sun.tests"]
package_dir = {name: name}

duplicate_package_info(packages, name, namespace)
duplicate_package_info(package_dir, name, namespace)

setup(
    name=name,
    author="Jean Connelly",
    description="Sun position calculator",
    author_email="jconnelly@cfa.harvard.edu",
    use_scm_version=True,
    setup_requires=["setuptools_scm", "setuptools_scm_git_archive"],
    zip_safe=False,
    package_dir=package_dir,
    packages=packages,
    tests_require=["pytest"],
    cmdclass=cmdclass,
)
