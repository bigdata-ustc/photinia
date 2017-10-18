#!/usr/bin/env python3


from setuptools import setup

if __name__ == '__main__':
    with open('README.rst') as file:
        long_description = file.read()
    setup(
        name='photinia',
        packages=['photinia'],
        version='0.1.20171018',
        keywords=('deep learning', 'neural network'),
        description='Build deep learning models quickly for scientists in an object-oriented way.',
        long_description=long_description,
        license='Free',
        author='dark_lab502',
        author_email='gylv@mail.ustc.edu.cn',
        url='https://github.com/XoriieInpottn/photinia',
        platforms='any',
        classifiers=[
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5'
        ],
        include_package_data=True,
        zip_safe=True
    )
