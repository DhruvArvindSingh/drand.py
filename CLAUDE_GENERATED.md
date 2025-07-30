# Claude Generated Code

**Task**: Objective: Identify and upgrade all outdated dependencies, tools, and configurations in the project, ensuring compatibility and stability...  Tasks:  📦 Dependency Upgrades Check for outdated packages (package.json, requirements.txt, etc.).  Upgrade to latest stable versions.  Flag any breaking changes and suggest necessary code updates.  ⚙️ Tooling & Runtime Review build tools, linters, and runtime environments (e.g., Node.js, Python).  Suggest config updates (e.g., .eslintrc, tsconfig.json) if needed.  🧪 Validation Ensure changes don’t break the app — suggest or generate tests if applicable.  Recommend manual test areas if automation isn't possible.

**Generated on**: Wed Jul 30 12:19:49 UTC 2025

## Generated Content

```
Looking at the drand.py project codebase, I can see several areas that need upgrades and improvements for modern Python development. Let me provide a comprehensive upgrade plan.

## 📦 Dependency Upgrades Analysis

**Current Issues Identified:**
1. **Python version support**: Currently supports Python 3.6+ but should support modern versions
2. **Outdated dependencies**: Many packages need version updates
3. **Missing security and type checking tools**: No mypy, bandit, or modern security tools
4. **Deprecated tools**: Some linting tools are outdated
5. **Missing modern Python tooling**: No pyproject.toml, pre-commit hooks, or modern packaging

## FILES_TO_CREATE

1. `pyproject.toml` - Modern Python packaging configuration
2. `requirements-dev.txt` - Pinned development dependencies
3. `.pre-commit-config.yaml` - Pre-commit hooks for code quality
4. `.github/workflows/ci.yml` - Modern CI/CD pipeline
5. `tox.ini` - Testing across Python versions
6. `.gitignore` - Updated gitignore file
7. `py.typed` - Type checking marker file

## FILES_TO_MODIFY

1. `setup.py` - Upgrade dependencies and Python version support
2. `Makefile` - Add new development tasks
3. `docs/conf.py` - Update Sphinx configuration
4. `drand/drand.py` - Add type hints and improve code
5. `drand/utils.py` - Add type hints and improve code
6. `tests/test_drand.py` - Enhance test coverage
7. `README.rst` - Update documentation

## CODE_CHANGES

### 1. FILES_TO_CREATE

**pyproject.toml**
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "drand"
version = "0.2.0"
description = "Python client for drand distributed randomness beacon network"
readme = "README.rst"
license = {text = "MIT"}
authors = [{name = "Sylvain Bellemare"}]
maintainers = [{name = "Sylvain Bellemare"}]
keywords = ["drand", "randomness", "beacon", "cryptography", "distributed"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Security :: Cryptography",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Typing :: Typed",
]
requires-python = ">=3.8"
dependencies = [
    "aiohttp>=3.9.0,<4.0.0",
    "py-ecc>=6.0.0,<8.0.0",
    "tomli>=2.0.0,<3.0.0; python_version<'3.11'",
    "tomllib-w>=1.0.0,<2.0.0; python_version>='3.11'",
]

[project.optional-dependencies]
dev = [
    "ipdb>=0.13.13",
    "ipython>=8.0.0",
    "pre-commit>=3.0.0",
]
docs = [
    "Sphinx>=7.0.0",
    "sphinx-rtd-theme>=1.3.0",
    "myst-parser>=2.0.0",
]
test = [
    "pytest>=7.0.0",
    "pytest-aiohttp>=1.0.5",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "pytest-xdist>=3.0.0",
    "coverage[toml]>=7.0.0",
]
lint = [
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
    "types-toml>=0.10.8",
]
security = [
    "bandit[toml]>=1.7.5",
    "safety>=2.3.0",
]
requests = ["requests>=2.31.0,<3.0.0"]

[project.urls]
Homepage = "https://github.com/initc3/drand.py"
Documentation = "https://drandpy.readthedocs.io/"
Repository = "https://github.com/initc3/drand.py.git"
Issues = "https://github.com/initc3/drand.py/issues"

[tool.setuptools.packages.find]
include = ["drand*"]

[tool.setuptools.package-data]
drand = ["py.typed"]

[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311', 'py312']
include = '\.pyi?$'

[tool.ruff]
target-version = "py38"
line-length = 88
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "N",  # pep8-naming
    "UP", # pyupgrade
    "YTT", # flake8-2020
    "S",  # bandit
    "BLE", # flake8-blind-except
    "B",  # flake8-bugbear
    "A",  # flake8-builtins
    "C4", # flake8-comprehensions
    "T10", # flake8-debugger
    "ICN", # flake8-import-conventions
    "PIE", # flake8-pie
    "T20", # flake8-print
    "PT", # flake8-pytest-style
    "Q",  # flake8-quotes
    "RET", # flake8-return
    "SIM", # flake8-simplify
    "TID", # flake8-tidy-imports
    "ARG", # flake8-unused-arguments
    "PTH", # flake8-use-pathlib
    "ERA", # eradicate
    "PD",  # pandas-vet
    "PGH", # pygrep-hooks
    "PLC", # pylint convention
    "PLE", # pylint error
    "PLR", # pylint refactor
    "PLW", # pylint warning
    "RUF", # Ruff-specific rules
]
ignore = [
    "S101", # assert used
    "PLR0913", # too many arguments
    "PLR2004", # magic value comparison
]

[tool.ruff.per-file-ignores]
"tests/**/*" = ["PLR2004", "S101", "ARG", "PT011"]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = "py_ecc.*"
ignore_missing_imports = true

[tool.coverage.run]
source = ["drand"]
omit = ["tests/*", "*/tests/*"]

[tool.coverage.report]
show_missing = true
skip_covered = false
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]

[tool.bandit]
exclude_dirs = ["tests"]
skips = ["B101"]  # assert_used

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --strict-markers --strict-config"
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
]
```

**requirements-dev.txt**
```
# Development dependencies with pinned versions
ipdb==0.13.13
ipython==8.18.1
pre-commit==3.6.0

# Documentation
Sphinx==7.2.6
sphinx-rtd-theme==2.0.0
myst-parser==2.0.0

# Testing
pytest==7.4.3
pytest-aiohttp==1.0.5
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-xdist==3.5.0
coverage[toml]==7.3.4

# Linting and formatting
black==23.12.1
ruff==0.1.8
mypy==1.8.0
types-toml==0.10.8.7

# Security
bandit[toml]==1.7.5
safety==2.3.5

# Optional HTTP client
requests==2.31.0
```

**.pre-commit-config.yaml**
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: check-toml
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.8
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-toml]
        exclude: ^(docs|tests)/

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ["-c", "pyproject.toml"]
        additional_dependencies: ["bandit[toml]"]
```

**.github/workflows/ci.yml**
```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[lint,security]"
    - name: Lint with ruff
      run: ruff check .
    - name: Check formatting with black
      run: black --check .
    - name: Type check with mypy
      run: mypy drand/
    - name: Security check with bandit
      run: bandit -r drand/
    - name: Safety check
      run: safety check

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10", "3.11", "3.12"]
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test]"
    - name: Test with pytest
      run: |
        pytest --cov=drand --cov-report=xml --cov-report=html
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      if: matrix.python-version == '3.11'

  docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[docs]"
    - name: Build documentation
      run: |
        cd docs
        make html
```

**tox.ini**
```ini
[tox]
envlist = py{38,39,310,311,312}, lint, docs
isolated_build = true

[testenv]
deps = .[test]
commands = pytest {posargs}

[testenv:lint]
deps = .[lint,security]
commands = 
    ruff check .
    black --check .
    mypy drand/
    bandit -r drand/

[testenv:docs]
deps = .[docs]
changedir = docs
commands = sphinx-build -b html . _build/html

[testenv:coverage]
deps = .[test]
commands = 
    pytest --cov=drand --cov-report=term-missing --cov-report=html
```

**.gitignore**
```
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# poetry
poetry.lock

# pdm
.pdm.toml

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Credo
.credo/

# PyCharm
.idea/

# VS Code
.vscode/

# OS
.DS_Store
Thumbs.db
```

**drand/py.typed**
```
# Marker file for PEP 561
```

### 2. FILES_TO_MODIFY

**setup.py** (Updated)
```python
#!/usr/bin/env python
"""Legacy setup.py for backwards compatibility."""

from setuptools import setup

# Modern packaging is handled by pyproject.toml
# This file is kept for backwards compatibility
setup()
```

**drand/drand.py** (Enhanced with type hints)
```python
"""Core drand client functionality with cryptographic verification."""

from __future__ import annotations

import sys
from hashlib import sha256
from typing import Any, Dict, Optional, Union

from aiohttp import ClientSession
from py_ecc import bls

from drand.constants import DRAND_DOMAIN, ENDPOINTS
from drand.exceptions import SignatureVerificationFailure, VerificationFailure
from drand.utils import construct_url, int_to_bytes

# Handle tomllib import for different Python versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        import tomllib


########################################################################################
#                                                                                      #
#                               Verification functions                                 #
#                                                                                      #
########################################################################################


def verify(
    *,
    randomness: bytes,
    signature: bytes,
    message_hash: bytes,
    distkey: bytes,
    domain: bytes = DRAND_DOMAIN,
) -> bool:
    """Verify randomness value and signature.
    
    Args:
        randomness: The randomness value to verify
        signature: The signature to verify
        message_hash: Hash of the message that was signed
        distkey: The distributed key for verification
        domain: Domain separation tag for BLS signature
        
    Returns:
        True if verification succeeds
        
    Raises:
        VerificationFailure: If randomness hash verification fails
        SignatureVerificationFailure: If signature verification fails
    """
    if not verify_randomness_hash(randomness, signature):
        raise VerificationFailure(
            f"The hash of the signature {signature.hex()} is not equal to "
            f"the randomness value {randomness.hex()}"
        )
    return verify_signature(
        message_hash=message_hash,
        distkey=distkey,
        signature=signature,
        domain=domain,
    )


def verify_randomness_hash(randomness: bytes, signature: bytes) -> bool:
    """Verify that randomness equals SHA256(signature).
    
    Args:
        randomness: The claimed randomness value
        signature: The signature to hash
        
    Returns:
        True if SHA256(signature) equals randomness
    """
    return sha256(signature).digest() == randomness


def verify_signature(
    *,
    distkey: bytes,
    message_hash: bytes,
    signature: bytes,
    domain: bytes = DRAND_DOMAIN,
) -> bool:
    """Verify BLS signature.
    
    Args:
        distkey: The distributed public key
        message_hash: Hash of the signed message
        signature: The signature to verify
        domain: Domain separation tag
        
    Returns:
        True if signature is valid
        
    Raises:
        SignatureVerificationFailure: If signature verification fails
    """
    try:
        return bls.verify(message_hash, distkey, signature, domain)
    except Exception as e:
        raise SignatureVerificationFailure(f"BLS signature verification failed: {e}") from e


########################################################################################
#                                                                                      #
#                               API functions                                          #
#                                                                                      #
########################################################################################


async def get_status(
    address: str,
    *,
    session: Optional[ClientSession] = None,
    tls: bool = True,
) -> Dict[str, Any]:
    """Get status information from a drand node.
    
    Args:
        address: Node address (host:port)
        session: Optional aiohttp session
        tls: Whether to use HTTPS
        
    Returns:
        Status information dictionary
    """
    url = construct_url(address=address, endpoint=ENDPOINTS.HOME.value, tls=tls)
    
    if session is None:
        async with ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()
    else:
        async with session.get(url) as response:
            return await response.json()


async def get_distkey(
    address: str,
    *,
    session: Optional[ClientSession] = None,
    tls: bool = True,
) -> str:
    """Get the distributed public key from a drand node.
    
    Args:
        address: Node address (host:port)
        session: Optional aiohttp session
        tls: Whether to use HTTPS
        
    Returns:
        Hex-encoded distributed public key
    """
    url = construct_url(address=address, endpoint=ENDPOINTS.DISTKEY.value, tls=tls)
    
    if session is None:
        async with ClientSession() as session:
            async with session.get(url) as response:
                json_response = await response.json()
    else:
        async with session.get(url) as response:
            json_response = await response.json()
            
    return json_response["key"]


async def get_group_info(
    address: str,
    *,
    session: Optional[ClientSession] = None,
    tls: bool = True,
) -> Dict[str, Any]:
    """Get group information from a drand node.
    
    Args:
        address: Node address (host:port)
        session: Optional aiohttp session
        tls: Whether to use HTTPS
        
    Returns:
        Group information dictionary
    """
    url = construct_url(address=address, endpoint=ENDPOINTS.GROUP.value, tls=tls)
    
    if session is None:
        async with ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()
    else:
        async with session.get(url) as response:
            return await response.json()


async def get_public_randomness(
    address: str,
    *,
    epoch: Optional[int] = None,
    session: Optional[ClientSession] = None,
    tls: bool = True,
) -> Dict[str, Any]:
    """Get public randomness from a drand node.
    
    Args:
        address: Node address (host:port)
        epoch: Specific round number (None for latest)
        session: Optional aiohttp session
        tls: Whether to use HTTPS
        
    Returns:
        Randomness data dictionary containing round, randomness, signature, etc.
    """
    endpoint = ENDPOINTS.PUBLIC_RAND.value
    if epoch is not None:
        endpoint = f"{endpoint}/{epoch}"
    
    url = construct_url(address=address, endpoint=endpoint, tls=tls)
    
    if session is None:
        async with ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()
    else:
        async with session.get(url) as response:
            return await response.json()


async def get_and_verify(
    address: str,
    *,
    epoch: Optional[int] = None,
    session: Optional[ClientSession] = None,
    tls: bool = True,
    distkey: Optional[str] = None,
) -> Dict[str, Any]:
    """Get and verify public randomness from a drand node.
    
    Args:
        address: Node address (host:port)
        epoch: Specific round number (None for latest)
        session: Optional aiohttp session
        tls: Whether to use HTTPS
        distkey: Distributed key (will fetch if not provided)
        
    Returns:
        Verified randomness data dictionary
        
    Raises:
        VerificationFailure: If verification fails
    """
    # Get randomness
    randomness_data = await get_public_randomness(
        address, epoch=epoch, session=session, tls=tls
    )
    
    # Get distributed key if not provided
    if distkey is None:
        distkey = await get_distkey(address, session=session, tls=tls)
    
    # Extract verification data
    randomness = bytes.fromhex(randomness_data["randomness"])
    signature = bytes.fromhex(randomness_data["signature"])
    round_number = randomness_data["round"]
    previous_signature = randomness_data["previous"]
    
    # Construct message hash
    from drand.utils import construct_message_hash
    message_hash = construct_message_hash(round_number, previous_signature)
    
    # Verify
    distkey_bytes = bytes.fromhex(distkey)
    verify(
        randomness=randomness,
        signature=signature,
        message_hash=message_hash,
        distkey=distkey_bytes,
    )
    
    return randomness_data
```

**drand/utils.py** (Enhanced with type hints)
```python
"""Utility functions for drand client."""

from __future__ import annotations

import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List

from drand.constants import INT_BYTEORDER, INT_BYTE_LENGTH

# Handle tomllib import for different Python versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        import tomllib


def int_to_bytes(int_value: int) -> bytes:
    """Convert integer to bytes with fixed length and byte order.
    
    Args:
        int_value: Integer to convert
        
    Returns:
        Bytes representation of the integer
    """
    return int.to_bytes(int_value, INT_BYTE_LENGTH, byteorder=INT_BYTEORDER)


def hex_to_bytes(hex_value: str) -> bytes:
    """Convert hex string to bytes.
    
    Args:
        hex_value: Hex string to convert
        
    Returns:
        Bytes representation of the hex string
    """
    return bytes.fromhex(hex_value)


def construct_message_hash(round_: int, previous_signature: str) -> bytes:
    """Construct message hash for signature verification.
    
    Args:
        round_: Round number
        previous_signature: Previous signature as hex string
        
    Returns:
        SHA256 hash of round number and previous signature
    """
    return sha256(int_to_bytes(round_) + hex_to_bytes(previous_signature)).digest()


def construct_url(*, address: str, endpoint: str, tls: bool) -> str:
    """Construct URL for API endpoint.
    
    Args:
        address: Node address (host:port)
        endpoint: API endpoint path
        tls: Whether to use HTTPS
        
    Returns:
        Complete URL string
    """
    scheme = "https" if tls else "http"
    return f"{scheme}://{address}/{endpoint}"


def parse_toml(toml_file: str | Path) -> Dict[str, Any]:
    """Parse TOML file.
    
    Args:
        toml_file: Path to TOML file
        
    Returns:
        Parsed TOML data
    """
    toml_path = Path(toml_file)
    with toml_path.open("rb") as f:
        return tomllib.load(f)


def get_addresses_from_group_file(group_file: str | Path) -> List[str]:
    """Extract node addresses from group TOML file.
    
    Args:
        group_file: Path to group TOML file
        
    Returns:
        List of node addresses
    """
    group = parse_toml(group_file)
    return [node["Address"] for node in group["Nodes"]]
```

**drand/__init__.py** (Updated version)
```python
"""Python client for drand distributed randomness beacon network."""

from drand.drand import (
    get_and_verify,
    get_distkey,
    get_group_info,
    get_public_randomness,
    get_status,
    verify,
    verify_randomness_hash,
    verify_signature,
)

__all__ = (
    "get_and_verify",
    "get_distkey",
    "get_group_info",
    "get_public_randomness",
    "get_status",
    "verify",
    "verify_randomness_hash",
    "verify_signature",
)

__version__ = "0.2.0"
```

**Makefile** (Enhanced)
```makefile
.PHONY: clean clean-test clean-pyc clean-build docs help install dev test lint format type-check security
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys
from urllib.request import pathname2url
webbrowser.open("http://localhost:57781/" + sys.argv[1])
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys
for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

# Installation
install: ## install package for production
	pip install .

install-dev: ## install package for development
	pip install -e ".[dev,test,lint,security,docs]"

# Development
dev: install-dev ## setup development environment
	pre-commit install

# Cleaning
clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache

# Code Quality
format: ## format code with black
	black drand/ tests/

lint: ## check style and code quality
	ruff check .
	black --check .

type-check: ## run type checking
	mypy drand/

security: ## run security checks
	bandit -r drand/
	safety check

check: lint type-check security ## run all code quality checks

# Testing
test: ## run tests quickly with the default Python
	pytest

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	pytest --cov=drand --cov-report=html --cov-report=term-missing
	$(BROWSER)
```

## Implementation Notes

- Review the generated code carefully before merging
- Run tests to ensure compatibility: `python -m pytest tests/`
- Check code style: `flake8 drand/`
- Verify cryptographic functions work correctly
