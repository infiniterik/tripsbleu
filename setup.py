from setuptools import setup, find_packages
from setuptools.command.install import install as _install

with open("README.md", "r") as fh:
    long_description = fh.read()

base = [
        "pytrips>=0.5.16",
        "Penman==1.2.1",
        "Levenshtein==0.18.1",
        "networkx==2.8.4"
    ]

if __name__ == '__main__':
    setup(
        name="tripsbleu",
        version="0.0.1",
        author="Rik Bose",
        author_email="rbose@cs.rochester.edu",
        description="BLEU score over semantic graphs with deep node similarity",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/mrmechko/tripsbleu",
        packages=find_packages(exclude=["test"]),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
            "Operating System :: OS Independent",
        ],
        install_requires=base
    )
