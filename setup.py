from setuptools import setup, find_packages

setup(
    name='package_sample',
    version='0.1',
    packages=find_packages(),
    description='Package Sample',
    long_description='Package Sample',
    auhtor='H. Watanabe',
    install_requires=[
        'numpy',
        'logging',
        'typing',
    ],
)
