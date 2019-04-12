from distutils.core import setup

setup(
    name='cic',
    version='0.1.0',
    author='Jurre H. Thiel',
    author_email='j.h.thiel@vu.nl',
    packages=['cic'],
    license='LICENSE',
    description='A library to compute the Changes-In-Changes estimator',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy >= 1.10.0',
        'joblib >= 0.13'
    ]
)
