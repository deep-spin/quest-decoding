import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fr:
    installation_requirements = fr.readlines()

setuptools.setup(
    name="quest-decoding",
    version="1.0.14",
    author="Goncalo Faria, Sweta Agrawal.",
    author_email="goncalofaria.research@gmail.com, swetaagrawal20@gmail.com",
    description="A package for sampling from intractable distributions with LLMs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deep-spin/quest-decoding",
    packages=setuptools.find_packages(),
    install_requires=installation_requirements,
    python_requires=">=3.8.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
