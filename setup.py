from setuptools import setup, find_packages

setup(
    name="probisim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "typer>=0.9.0",
        "graphviz>=0.20.1",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "streamlit>=1.22.0",
    ],
    entry_points={
        "console_scripts": [
            "probisim=probisim.cli:app",
        ],
    },
    python_requires=">=3.8",
) 