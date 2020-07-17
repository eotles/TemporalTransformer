import setuptools


#with open("README.md", "r") as fh:
#    long_description = fh.read()

setuptools.setup(
    name="TemporalTransformer",
    version="0.0.1",
    author="Erkin Ötleş",
    author_email="eotles@gmail.com",
    description="Data transformation package for dynamic machine learning",
#    long_description=long_description,
    long_description="fuck",
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
