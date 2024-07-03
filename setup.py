import os
import pkg_resources
from setuptools import setup, find_packages


requirements = []


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="pacgoc",
    version="0.1.0",
    author="Imiloin",
    url="https://github.com/Imiloin/pacgoc",
    license="MIT",
    description="An audio analysis toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "app"]),
    python_requires=">=3.10",
    install_requires=requirements
    + [
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)
