from setuptools import setup, find_packages

setup(
    name="openbtk",
    version="0.1.0",
    description="A multi-modal biomedical vector database ingestion and processing library.",
    author="OpenBTK Team",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "pydantic",
        "pillow",
        "numpy",
        "librosa",
        # Add other dependencies as needed
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
