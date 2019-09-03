#!/usr/bin/env python3
import os

from setuptools import setup


def find_packages(dir_):
    _packages = []
    for filename in os.listdir(dir_):
        file_ = os.path.join(dir_, filename)
        if os.path.isdir(file_):
            _packages.extend(find_packages(file_))
        if '__init__.py' == filename:
            _packages.append(dir_.replace('/', '.'))
    return _packages


if __name__ == '__main__':
    packages = find_packages('photinia')
    packages.sort()
    with open('README.md') as file:
        long_description = file.read()
    setup(
        name='photinia',
        packages=packages,
        version='0.6.4.20190821',
        keywords=('deep learning', 'neural network'),
        description='Build deep learning models quickly for scientists in an object-oriented way.',
        long_description=long_description,
        license='Free',
        author='darklab_502',
        author_email='gylv@mail.ustc.edu.cn',
        url='https://github.com/XoriieInpottn/photinia',
        platforms='any',
        classifiers=[
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        include_package_data=True,
        zip_safe=True,
        install_requires=[
            'numpy',
            'scipy',
            'pymongo',
            'opencv-python',
            'prettytable'
        ]
    )
