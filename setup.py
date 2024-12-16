"""
Setup configuration for the Amazon Reviews Analysis package.

This package provides tools for analyzing Amazon product reviews using
machine learning techniques, including sentiment analysis, category clustering,
and review summarization.

Requirements are specified in environment.yml and individual requirement files
in the requirements directory.
"""

from setuptools import setup, find_packages

setup(
    name="amazon_reviews",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[],
    extras_require={
        'dev': []
    }
)
