import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyhank",
    version="2.2.1",
    author="Edward Rogers",
    author_email="etfrogers@hotmail.com",
    description="pyhank - Quasi-discrete Hankel transforms for python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/etfrogers/pyhank",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='~=3.6',
    install_requires=[
        'numpy~=1.15',
        'scipy~=1.0',
    ]
)
