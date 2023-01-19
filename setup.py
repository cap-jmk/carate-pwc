import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(
    name="carate",
    version="0.1.3",
    author="Julian M. Kleber",
    author_email="julian.kleber@sail.black",
    description="Graph-based encoder algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://www.codeberg/sail.black/carate",
    packages=setuptools.find_packages(include=["carate*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=["black", "Click"],
    entry_points={
        "console_scripts": [
            "carate = carate.automate:train_algorithm",
        ],
    },
    python_requires=">=3.9",
)
