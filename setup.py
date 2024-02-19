from typing import Dict, List
from setuptools import setup, find_packages

VERSION: Dict[str, str] = {}

with open("text_machina/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

with open("README.md", "r", encoding="utf-8") as readme:
    long_description = readme.read()

TEXT_GENERATION_DEPS = [
    "scipy>=1.10.1",
    "PyYAML>=6.0.1",
    "datasets>=2.14.4",
    "spacy>=3.6.1",
    "typer>=0.9.0",
    "pydantic>=2.3.0",
    "petname>=2.6",
    "pycountry>=22.3.5",
    "ftfy>=6.1.3",
    "fasttext-wheel",
]

EXPLORE_DEPS = [
    "rich>=13.7.0",
    "scikit-learn>=1.3.2",
    "mauve-text>=0.3.0",
    "matplotlib>=3.7.4",
    "tabulate>=0.9.0",
    "readchar>=4.0.5",
    "evaluate>=0.4.1",
    "textstat>=0.7.3",
    "seqeval>=1.2.2",
]

DEPS = TEXT_GENERATION_DEPS + EXPLORE_DEPS

DEV_DEPS = [
    "black",
    "flake8",
    "mypy",
    "types-requests",
    "pytest",
    "isort",
    "autoflake",
    "pre-commit",
    "pytest-sphinx",
    "pytest-cov",
    "ruff",
    "Sphinx>=4.3.0,<7.1.0",
    "furo==2023.7.26",
    "myst-parser>=1.0,<2.1",
    "sphinx-copybutton==0.5.2",
    "sphinx-autobuild==2021.3.14",
    "sphinx-autodoc-typehints==1.23.3",
    "packaging",
    "setuptools",
    "build",
    "wheel",
]

EXTRAS_REQUIRES: Dict[str, List[str]] = {
    "openai": ["openai>=1", "tiktoken>=0.4.0"],
    "azure_openai": ["openai>=1", "tiktoken>=0.4.0"],
    "bedrock": ["boto3", "tiktoken>=0.4.0"],
    "ai21": ["ai21>=2.0.0", "ai21_tokenizer>=0.3.11"],
    "anthropic": ["anthropic>=0.7.2"],
    "cohere": ["cohere>=4.36"],
    "huggingface-local": [
        "torch==2.0.1",
        "transformers>=4.32.0",
        "accelerate>=0.22.0",
        "bitsandbytes>=0.41.1",
    ],
    "huggingface-remote": ["requests>=2.31.0"],
    "vertex": ["google-auth", "google-cloud-aiplatform==1.25.0", "tiktoken>=0.4.0"],
}
EXTRAS_REQUIRES["all"] = sum(EXTRAS_REQUIRES.values(), [])
EXTRAS_REQUIRES["dev"] = EXTRAS_REQUIRES["all"] + DEV_DEPS

setup(
    version=VERSION["VERSION"],
    name="text-machina",
    description="Text Machina: Seamless Generation of Machine-Generated Text Datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Genaios",
    entry_points={"console_scripts": ["text-machina=text_machina.cli:app"]},
    packages=find_packages(),
    install_requires=DEPS,
    include_package_data=True,
    python_requires=">=3.8.0",
    extras_require=EXTRAS_REQUIRES,
    url="https://github.com/Genaios/TextMachina",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
    ],
    license_files=[
        "LICENSE",
    ],
)
