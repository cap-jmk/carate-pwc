import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(
    name="carate",
    version="0.1.0",
    author="Julian M. Kleber",
    author_email="julian.kleber@sail.black",
    description="Graph-based encoder algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://www.codeberg/sail.black/serial-sphinx",
    packages=setuptools.find_packages(include=["serial_sphinx"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    python_requires=">=3.9",
)
