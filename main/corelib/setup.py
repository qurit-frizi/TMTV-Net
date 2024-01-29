import re
from codecs import open
from os import path

from setuptools import find_packages, setup

import setup_utils

# customize library name here
NAME = setup_utils.get_project_name()

META_PATH = path.join('src', NAME, 'metadata.py')

def find_meta(meta):
    """
    Extract __*meta*__ from META_FILE.
    """
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
        META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')
    
with open(META_PATH, encoding='utf-8') as f:
    META_FILE = f.read()

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name=find_meta('name'),
    version=find_meta('version'),
    description=find_meta('description'),
    long_description=long_description,
    url=find_meta('url'),
    download_url=find_meta('url') + '/tarball/' + find_meta('version'),
    license=find_meta('license'),
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
    ],
    keywords='',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    author=find_meta('author'),
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email=find_meta('email')
)
