from __future__ import print_function
import os

from setuptools import setup, find_packages

import action_optimizer

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

def get_reqs(*fns):
    lst = []
    for fn in fns:
        for package in open(os.path.join(CURRENT_DIR, fn)).readlines():
            package = package.strip()
            if not package:
                continue
            lst.append(package.strip())
    return lst

setup(
    name="action_optimizer",
    version=action_optimizer.__version__,
    packages=find_packages(),
    package_data={
        'action_optimizer': [
            'fixtures/*',
        ],
    },
    author="Chris Spencer",
    author_email="chrisspen@gmail.com",
    description="A tool for analyzing sequence metrics to find an optimal daily routine.",
    license="BSD",
    url="https://github.com/chrisspen/action-optimizer",
    #https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Framework :: Django',
    ],
    zip_safe=False,
    install_requires=get_reqs('requirements.txt'),
    tests_require=get_reqs('requirements-test.txt'),
)
