import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="paysage",
    version="1.02",
    author="Vladislav Averett",
    author_email="avrtt@tuta.io",
    description="Pandas extras: find data quality issues and clean/improve dataframes in one line using scikit-learn transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    url="https://github.com/avrtt/paysage",
    py_modules = ["paysage"],
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=[
        "numpy>=1.21.5",
        "pandas>=1.3.5",
        "scikit-learn>=0.24.2",
    ],
    include_package_data = True,
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
)
