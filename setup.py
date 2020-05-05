"""
Module configuration.
"""

from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='mliis',
    version='0.0.1',
    description='Meta-learning initializations for image segmentation',
    long_description='Code for reproducing experiments in https://arxiv.org/abs/1912.06290',
    url='https://github.com/ml4ai/mliis',
    author='Sean M. Hendryx',
    author_email='seanmhendryx@email.arizona.edu',
    license='MIT',
    keywords='meta-learning image segmentation ai machine learning',
    packages=['mliis'],
    install_requires=requirements
)
