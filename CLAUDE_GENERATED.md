# Claude Generated Code

**Task**: Objective: Audit the entire codebase and identify improvements in:  🔐 Security Detect insecure patterns (e.g. unvalidated inputs, unsafe data handling).  Suggest best practices (e.g. input sanitization, environment variable usage).  🛠️ Maintainability Flag complex or duplicate code, unclear logic, poor naming.  Recommend modularization and better structure.  🚀 Performance Spot inefficient loops, redundant logic, unnecessary API calls.  Suggest caching or optimization where needed.  🧹 Clean-up Find dead code, unused variables/functions/imports.  List outdated or unnecessary dependencies.

**Generated on**: Wed Jul 30 12:17:17 UTC 2025

## Generated Content

```
I'll audit the entire codebase and provide comprehensive improvements across security, maintainability, performance, and clean-up. Let me analyze each aspect systematically.

## 🔍 AUDIT FINDINGS & IMPROVEMENTS

### FILES_TO_CREATE

1. `drand/validators.py` - Input validation utilities
2. `drand/config.py` - Configuration management
3. `drand/types.py` - Type definitions
4. `requirements.txt` - Explicit dependency management
5. `tests/test_validators.py` - Validator tests
6. `tests/test_utils.py` - Utils tests
7. `.env.example` - Environment variable template

### FILES_TO_MODIFY

1. `setup.py` - Security and dependency updates
2. `drand/drand.py` - Security, performance, and maintainability improvements
3. `drand/utils.py` - Enhanced error handling and validation
4. `drand/exceptions.py` - Additional exception types
5. `drand/constants.py` - Better configuration management
6. `drand/__init__.py` - Updated exports
7. `tests/test_drand.py` - Enhanced test coverage

## CODE_CHANGES

### FILES_TO_CREATE

#### `drand/validators.py`
```python
"""Input validation utilities for drand client."""

import re
from typing import Any, Optional
from urllib.parse import urlparse

from drand.exceptions import ValidationError


def validate_address(address: str) -> str:
    """Validate and normalize network address.
    
    Args:
        address: Network address (host:port or just host)
        
    Returns:
        Normalized address
        
    Raises:
        ValidationError: If address format is invalid
    """
    if not address or not isinstance(address, str):
        raise ValidationError("Address must be a non-empty string")
    
    address = address.strip()
    
    # Basic format validation
    if not re.match(r'^[a-zA-Z0-9.-]+(?::[0-9]+)?$', address):
        raise ValidationError(f"Invalid address format: {address}")
    
    # Check for localhost/private IPs in production (could be configurable)
    if address.startswith(('127.', '10.', '192.168.', 'localhost')):
        # Log warning but don't fail - might be legitimate in dev/test
        pass
    
    return address


def validate_url(url: str) -> str:
    """Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        Validated URL
        
    Raises:
        ValidationError: If URL format is invalid
    """
    if not url or not isinstance(url, str):
        raise ValidationError("URL must be a non-empty string")
    
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            raise ValidationError(f"Invalid URL format: {url}")
        
        if parsed.scheme not in ('http', 'https'):
            raise ValidationError(f"Unsupported URL scheme: {parsed.scheme}")
            
        return url
    except Exception as e:
        raise ValidationError(f"URL validation failed: {e}")


def validate_hex_string(hex_str: str, expected_length: Optional[int] = None) -> str:
    """Validate hexadecimal string.
    
    Args:
        hex_str: Hex string to validate
        expected_length: Expected byte length (None to skip length check)
        
    Returns:
        Validated hex string (lowercase)
        
    Raises:
        ValidationError: If hex string is invalid
    """
    if not isinstance(hex_str, str):
        raise ValidationError("Hex string must be a string")
    
    hex_str = hex_str.strip().lower()
    
    if not re.match(r'^[0-9a-f]*$', hex_str):
        raise ValidationError("Invalid hex string format")
    
    if len(hex_str) % 2 != 0:
        raise ValidationError("Hex string must have even length")
    
    if expected_length is not None:
        actual_length = len(hex_str) // 2
        if actual_length != expected_length:
            raise ValidationError(
                f"Expected {expected_length} bytes, got {actual_length} bytes"
            )
    
    return hex_str


def validate_round_number(round_num: Any) -> int:
    """Validate round number.
    
    Args:
        round_num: Round number to validate
        
    Returns:
        Validated round number
        
    Raises:
        ValidationError: If round number is invalid
    """
    if not isinstance(round_num, (int, str)):
        raise ValidationError("Round number must be integer or string")
    
    try:
        round_int = int(round_num)
        if round_int < 0:
            raise ValidationError("Round number must be non-negative")
        return round_int
    except ValueError:
        raise ValidationError(f"Invalid round number: {round_num}")


def sanitize_user_input(data: Any, max_length: int = 1000) -> str:
    """Sanitize user input for logging/display.
    
    Args:
        data: Input data to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string representation
    """
    if data is None:
        return "None"
    
    sanitized = str(data)[:max_length]
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[^\w\s.-]', '', sanitized)
    
    if len(str(data)) > max_length:
        sanitized += "...[truncated]"
    
    return sanitized
```

#### `drand/config.py`
```python
"""Configuration management for drand client."""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

from drand.constants import DRAND_DOMAIN


@dataclass
class DrandConfig:
    """Configuration for drand client."""
    
    # Network settings
    default_timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Security settings
    verify_ssl: bool = True
    max_response_size: int = 1024 * 1024  # 1MB
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    
    # Domain for BLS verification
    domain: bytes = DRAND_DOMAIN
    
    @classmethod
    def from_env(cls) -> 'DrandConfig':
        """Create configuration from environment variables."""
        return cls(
            default_timeout=float(os.getenv('DRAND_TIMEOUT', 30.0)),
            max_retries=int(os.getenv('DRAND_MAX_RETRIES', 3)),
            retry_delay=float(os.getenv('DRAND_RETRY_DELAY', 1.0)),
            verify_ssl=os.getenv('DRAND_VERIFY_SSL', 'true').lower() == 'true',
            max_response_size=int(os.getenv('DRAND_MAX_RESPONSE_SIZE', 1024 * 1024)),
            enable_caching=os.getenv('DRAND_ENABLE_CACHING', 'true').lower() == 'true',
            cache_ttl=int(os.getenv('DRAND_CACHE_TTL', 300)),
        )


# Global configuration instance
config = DrandConfig.from_env()
```

#### `drand/types.py`
```python
"""Type definitions for drand client."""

from typing import TypedDict, Optional


class RandomnessResponse(TypedDict):
    """Type for randomness API response."""
    round: int
    previous: str
    signature: str
    randomness: str


class DistKeyResponse(TypedDict):
    """Type for distkey API response."""
    key: str


class GroupInfo(TypedDict):
    """Type for group info response."""
    nodes: list
    threshold: int
    period: str
    genesis_time: str
    transition_time: str
    genesis_seed: str


class NodeInfo(TypedDict):
    """Type for node information."""
    Address: str
    Key: str
    TLS: bool
```

#### `requirements.txt`
```txt
aiohttp>=3.8.0,<4.0.0
py_ecc>=6.0.0,<7.0.0
toml>=0.10.0
```

#### `.env.example`
```bash
# Drand client configuration
DRAND_TIMEOUT=30.0
DRAND_MAX_RETRIES=3
DRAND_RETRY_DELAY=1.0
DRAND_VERIFY_SSL=true
DRAND_MAX_RESPONSE_SIZE=1048576
DRAND_ENABLE_CACHING=true
DRAND_CACHE_TTL=300
```

### FILES_TO_MODIFY

#### `setup.py` (Enhanced security and dependencies)
```python
#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.rst", encoding="utf-8") as readme_file:
    readme = readme_file.read()

# Pin dependencies for security
install_requires = [
    "aiohttp>=3.8.0,<4.0.0",  # Updated for security patches
    "py_ecc>=6.0.0,<7.0.0",   # Updated, more specific version
    "toml>=0.10.0,<1.0.0",    # More specific version constraint
]

dev_requires = ["ipdb", "ipython"]
docs_requires = ["Sphinx>=4.0.0"]
tests_requires = [
    "black>=22.0.0",
    "coverage>=6.0.0",
    "flake8>=4.0.0",
    "flake8-import-order>=0.18.0",
    "pep8-naming>=0.13.0",
    "pytest>=7.0.0",
    "pytest-aiohttp>=1.0.0",
    "pytest-cov>=3.0.0",
    "pytest-asyncio>=0.20.0",
    "mypy>=0.991",
]

extras_require = {
    "dev": dev_requires + docs_requires + tests_requires,
    "docs": docs_requires,
    "tests": tests_requires,
}

setup(
    author="Sylvain Bellemare",
    python_requires=">=3.7",  # Updated minimum version
    classifiers=[
        "Development Status :: 3 - Alpha",  # Updated status
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description="Python client for drand distributed randomness beacon.",
    install_requires=install_requires,
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/x-rst",
    include_package_data=True,
    keywords="drand,randomness,beacon,cryptography,distributed",
    name="drand",
    packages=find_packages(include=["drand", "drand.*"]),
    url="https://github.com/initc3/drand.py",
    version="0.2.0",  # Version bump for security improvements
    zip_safe=False,
    extras_require=extras_require,
)
```

#### `drand/exceptions.py` (Additional exception types)
```python
"""Exception classes for drand client."""


class DrandException(Exception):
    """Base class for drand exceptions."""
    pass


class VerificationFailure(DrandException):
    """Error raised when the verification for a random value fails."""
    pass


class SignatureVerificationFailure(VerificationFailure):
    """Error raised when the verification of the signature fails."""
    pass


class ValidationError(DrandException):
    """Error raised when input validation fails."""
    pass


class NetworkError(DrandException):
    """Error raised when network operations fail."""
    pass


class ConfigurationError(DrandException):
    """Error raised when configuration is invalid."""
    pass


class TimeoutError(NetworkError):
    """Error raised when operations timeout."""
    pass


class ResponseError(NetworkError):
    """Error raised when API response is invalid."""
    pass
```

#### `drand/constants.py` (Enhanced configuration)
```python
"""Constants for drand client."""

from enum import Enum


# BLS domain separation tag for drand
DRAND_DOMAIN = bytes([1, 9, 6, 9, 9, 6, 9, 2])

# Integer serialization constants
INT_BYTE_LENGTH = 8
INT_BYTEORDER = "big"

# Network constants
DEFAULT_TIMEOUT = 30.0
MAX_RESPONSE_SIZE = 1024 * 1024  # 1MB
DEFAULT_RETRIES = 3

# Cache constants
DEFAULT_CACHE_TTL = 300  # 5 minutes


class ENDPOINTS(Enum):
    """API endpoints for drand service."""
    HOME = "api"
    PUBLIC_RAND = "api/public"
    DISTKEY = "api/info/distkey"
    GROUP = "api/info/group"


# Expected response sizes for validation
EXPECTED_SIGNATURE_LENGTH = 96  # bytes
EXPECTED_RANDOMNESS_LENGTH = 32  # bytes
EXPECTED_DISTKEY_LENGTH = 96    # bytes (compressed G2 point)
```

#### `drand/utils.py` (Enhanced with validation and error handling)
```python
"""Utility functions for drand client."""

from hashlib import sha256
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

import toml

from drand.constants import INT_BYTEORDER, INT_BYTE_LENGTH
from drand.exceptions import ValidationError, ConfigurationError
from drand.validators import validate_hex_string, validate_address

logger = logging.getLogger(__name__)


def int_to_bytes(int_value: int) -> bytes:
    """Convert integer to bytes with validation.
    
    Args:
        int_value: Integer to convert
        
    Returns:
        Bytes representation
        
    Raises:
        ValidationError: If integer is invalid
    """
    if not isinstance(int_value, int):
        raise ValidationError("Value must be an integer")
    
    if int_value < 0:
        raise ValidationError("Value must be non-negative")
    
    try:
        return int.to_bytes(int_value, INT_BYTE_LENGTH, byteorder=INT_BYTEORDER)
    except OverflowError:
        raise ValidationError(f"Integer {int_value} too large for {INT_BYTE_LENGTH} bytes")


def hex_to_bytes(hex_value: str) -> bytes:
    """Convert hex string to bytes with validation.
    
    Args:
        hex_value: Hex string to convert
        
    Returns:
        Bytes representation
        
    Raises:
        ValidationError: If hex string is invalid
    """
    validated_hex = validate_hex_string(hex_value)
    
    try:
        return bytes.fromhex(validated_hex)
    except ValueError as e:
        raise ValidationError(f"Failed to convert hex to bytes: {e}")


def construct_message_hash(round_: int, previous_signature: str) -> bytes:
    """Construct message hash for signature verification.
    
    Args:
        round_: Round number
        previous_signature: Previous signature as hex string
        
    Returns:
        Message hash
        
    Raises:
        ValidationError: If inputs are invalid
    """
    if not isinstance(round_, int) or round_ < 0:
        raise ValidationError("Round must be a non-negative integer")
    
    try:
        round_bytes = int_to_bytes(round_)
        previous_bytes = hex_to_bytes(previous_signature)
        return sha256(round_bytes + previous_bytes).digest()
    except Exception as e:
        logger.error(f"Failed to construct message hash: {e}")
        raise ValidationError(f"Failed to construct message hash: {e}")


def construct_url(*, address: str, endpoint: str, tls: bool) -> str:
    """Construct URL with validation.
    
    Args:
        address: Network address
        endpoint: API endpoint
        tls: Whether to use HTTPS
        
    Returns:
        Complete URL
        
    Raises:
        ValidationError: If inputs are invalid
    """
    validated_address = validate_address(address)
    
    if not isinstance(endpoint, str) or not endpoint:
        raise ValidationError("Endpoint must be a non-empty string")
    
    scheme = "https" if tls else "http"
    url = f"{scheme}://{validated_address}/{endpoint.lstrip('/')}"
    
    logger.debug(f"Constructed URL: {url}")
    return url


def parse_toml(toml_file: str) -> Dict[str, Any]:
    """Parse TOML file with error handling.
    
    Args:
        toml_file: Path to TOML file
        
    Returns:
        Parsed TOML data
        
    Raises:
        ConfigurationError: If file cannot be parsed
    """
    try:
        file_path = Path(toml_file)
        if not file_path.exists():
            raise ConfigurationError(f"TOML file does not exist: {toml_file}")
        
        if not file_path.is_file():
            raise ConfigurationError(f"Path is not a file: {toml_file}")
        
        content = file_path.read_text(encoding='utf-8')
        return toml.loads(content)
        
    except toml.TomlDecodeError as e:
        raise ConfigurationError(f"Failed to parse TOML file {toml_file}: {e}")
    except (OSError, IOError) as e:
        raise ConfigurationError(f"Failed to read TOML file {toml_file}: {e}")


def get_addresses_from_group_file(group_file: str) -> List[str]:
    """Extract addresses from group file with validation.
    
    Args:
        group_file: Path to group file
        
    Returns:
        List of node addresses
        
    Raises:
        ConfigurationError: If group file is invalid
    """
    try:
        group = parse_toml(group_file)
        
        if 'Nodes' not in group:
            raise ConfigurationError("Group file missing 'Nodes' section")
        
        nodes = group['Nodes']
        if not isinstance(nodes, list):
            raise ConfigurationError("'Nodes' must be a list")
        
        addresses = []
        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                raise ConfigurationError(f"Node {i} must be a dictionary")
            
            if 'Address' not in node:
                raise ConfigurationError(f"Node {i} missing 'Address' field")
            
            address = validate_address(node['Address'])
            addresses.append(address)
        
        return addresses
        
    except Exception as e:
        logger.error(f"Failed to extract addresses from group file: {e}")
        raise


def safe_hex_display(hex_value: str, max_length: int = 16) -> str:
    """Safely display hex value for logging (truncated).
    
    Args:
        hex_value: Hex string to display
        max_length: Maximum characters to show
        
    Returns:
        Truncated hex string for safe display
    """
    if not isinstance(hex_value, str):
        return "invalid"
    
    if len(hex_value) <= max_length:
        return hex_value
    
    return f"{hex_value[:max_length]}...({len(hex_value)} chars)"
```

#### `drand/drand.py` (Comprehensive improvements)
```python
"""Main drand client implementation with security and performance improvements."""

import asyncio
import logging
from hashlib import sha256
from typing import Optional, Dict, Any, Union
import time

from aiohttp import ClientSession, ClientTimeout, ClientError

from py_ecc import bls

from drand.constants import (
    DRAND_DOMAIN, 
    ENDPOINTS, 
    DEFAULT_TIMEOUT,
    MAX_RESPONSE_SIZE,
    DEFAULT_RETRIES,
    EXPECTED_SIGNATURE_LENGTH,
    EXPECTED_RANDOMNESS_LENGTH,
    EXPECTED_DISTKEY_LENGTH,
)
from drand.exceptions import (
    SignatureVerificationFailure, 
    VerificationFailure,
    NetworkError,
    TimeoutError,
    ResponseError,
    ValidationError,
)
from drand.utils import construct_url, hex_to_bytes, safe_hex_display
from drand.validators import validate_address, validate_round_number, validate_hex_string
from drand.config import config
from drand.types import RandomnessResponse, DistKeyResponse

logger = logging.getLogger(__name__)

# Simple in-memory cache for performance
_cache: Dict[str, Dict[str, Any]] = {}


def _cache_key(prefix: str, *args) -> str:
    """Generate cache key."""
    return f"{prefix}:{':'.join(str(arg) for arg in args)}"


def _cache_get(key: str) -> Optional[Any]:
    """Get from cache if not expired."""
    if not config.enable_caching:
        return None
    
    entry = _cache.get(key)
    if entry and time.time() - entry['timestamp'] < config.cache_ttl:
        logger.debug(f"Cache hit for key: {key}")
        return entry['data']
    
    # Remove expired entry
    if entry:
        _cache.pop(key, None)
        logger.debug(f"Cache expired for key: {key}")
    
    return None


def _cache_set(key: str, data: Any) -> None:
    """Set cache entry."""
    if config.enable_caching:
        _cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
        logger.debug(f"Cached data for key: {key}")


########################################################################################
#                                Verification functions                                 #
########################################################################################

def verify(
    *, 
    randomness: Union[str, bytes], 
    signature: Union[str, bytes], 
    message_hash: bytes, 
    distkey: Union[str, bytes], 
    domain: bytes = DRAND_DOMAIN
) -> bool:
    """Verify drand randomness with comprehensive validation.
    
    Args:
        randomness: Random value (hex string or bytes)
        signature: BLS signature (hex string or bytes)
        message_hash: Message hash for verification
        distkey: Distributed public key (hex string or bytes)
        domain: BLS domain for verification
        
    Returns:
        True if verification successful
        
    Raises:
        VerificationFailure: If verification fails
        ValidationError: If inputs are invalid
    """
    try:
        # Convert inputs to bytes with validation
        if isinstance(randomness, str):
            randomness_bytes = hex_to_bytes(validate_hex_string(randomness, EXPECTED_RANDOMNESS_LENGTH))
        else:
            randomness_bytes = randomness
        
        if isinstance(signature, str):
            signature_bytes = hex_to_bytes(validate_hex_string(signature, EXPECTED_SIGNATURE_LENGTH))
        else:
            signature_bytes = signature
            
        if isinstance(distkey, str):
            distkey_bytes = hex_to_bytes(validate_hex_string(distkey, EXPECTED_DISTKEY_LENGTH))
        else:
            distkey_bytes = distkey
        
        # Verify randomness hash first
        if not verify_randomness_hash(randomness_bytes, signature_bytes):
            raise VerificationFailure(
                f"Hash of signature {safe_hex_display(signature_bytes.hex())} "
                f"does not match randomness {safe_hex_display(randomness_bytes.hex())}"
            )
        
        # Verify BLS signature
        return verify_signature(
            message_hash=message_hash,
            distkey=distkey_bytes,
            signature=signature_bytes,
            domain=domain,
        )
        
    except Exception as e:
        if isinstance(e, (VerificationFailure, ValidationError)):
            raise
        logger.error(f"Unexpected error in verification: {e}")
        raise VerificationFailure(f"Verification failed: {e}")


def verify_randomness_hash(randomness: bytes, signature: bytes) -> bool:
    """Verify that randomness is SHA256 hash of signature.
    
    Args:
        randomness: Random value bytes
        signature: Signature bytes
        
    Returns:
        True if randomness equals SHA256(signature)
    """
    try:
        computed_randomness = sha256(signature).digest()
        result = computed_randomness == randomness
        
        if not result:
            logger.warning(
                f"Randomness hash mismatch: "
                f"expected {safe_hex_display(randomness.hex())}, "
                f"got {safe_hex_display(computed_randomness.hex())}"
            )
        
        return result
    except Exception as e:
        logger.error(f"Error verifying randomness hash: {e}")
        return False


def verify_signature(
    *, 
    distkey: bytes, 
    message_hash: bytes, 
    signature: bytes, 
    domain: bytes = DRAND_DOMAIN
) -> bool:
    """Verify BLS signature with error handling.
    
    Args:
        distkey: Distributed public key bytes
        message_hash: Message hash bytes
        signature: Signature bytes
        domain: BLS domain bytes
        
    Returns:
        True if signature is valid
        
    Raises:
        SignatureVerificationFailure: If signature verification fails
    """
    try:
        result = bls.verify(message_hash, distkey, signature, domain)
        
        if not result:
            raise SignatureVerificationFailure(
                f"BLS signature verification failed for message "
                f"{safe_hex_display(message_hash.hex())}"
            )
        
        logger.debug("BLS signature verification successful")
        return result
        
    except Exception as e:
        if isinstance(e, SignatureVerificationFailure):
            raise
        logger.error(f"Error in BLS signature verification: {e}")
        raise SignatureVerificationFailure(f"Signature verification error: {e}")


########################################################################################
#                                Network functions                                      #
########################################################################################

async def _make_request(
    url: str, 
    session: Optional[ClientSession] = None,
    timeout: Optional[float] = None,
    max_retries: int = DEFAULT_RETRIES
) -> Dict[str, Any]:
    """Make HTTP request with retry logic and validation.
    
    Args:
        url: URL to request
        session: Optional aiohttp session
        timeout: Request timeout
        max_retries: Maximum retry attempts
        
    Returns:
        JSON response data
        
    Raises:
        NetworkError: If request fails
        TimeoutError: If request times out
        ResponseError: If response is invalid
    """
    timeout_val = timeout or config.default_timeout
    client_timeout = ClientTimeout(total=timeout_val)
    
    session_provided = session is not None
    if not session_provided:
        session = ClientSession(timeout=client_timeout)
    
    last_error = None
    
    try:
        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"Making request to {url} (attempt {attempt + 1})")
                
                async with session.get(
                    url, 
                    ssl=config.verify_ssl,
                    timeout=client_timeout
                ) as response:
                    # Validate response size
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > config.max_response_size:
                        raise ResponseError(f"Response too large: {content_length} bytes")
                    
                    # Read response with size limit
                    content = await response.read()
                    if len(content) > config.max_response_size:
                        raise ResponseError(f"Response too large: {len(content)} bytes")
                    
                    if response.status != 200:
                        raise NetworkError(f"HTTP {response.status}: {response.reason}")
                    
                    try:
                        data = await response.json()
                        logger.debug(f"Successful response from {url}")
                        return data
                    except Exception as e:
                        raise ResponseError(f"Invalid JSON response: {e}")
                        
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Request to {url} timed out after {timeout_val}s")
                logger.warning(f"Request timeout (attempt {attempt + 1}): {url}")
                
            except ClientError as e:
                last_error = NetworkError(f"Network error: {e}")
                logger.warning(f"Network error (attempt {attempt + 1}): {e}")
                
            except Exception as e:
                last_error = NetworkError(f"Unexpected error: {e}")
                logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")
            
            # Wait before retry
            if attempt < max_retries:
                wait_time = config.retry_delay * (2 ** attempt)  # Exponential backoff
                logger.debug(f"Waiting {wait_time}s before retry")
                await asyncio.sleep(wait_time)
    
    finally:
        if not session_provided and session:
            await session.close()
    
    # All retries failed
    raise last_error or NetworkError("All retry attempts failed")


async def get_distkey(
    address: str, 
    *, 
    session: Optional[ClientSession] = None, 
    tls: bool = True,
    timeout: Optional[float] = None
) -> str:
    """Get distributed public key from drand node.
    
    Args:
        address: Node address
        session: Optional aiohttp session
        tls: Use HTTPS if True
        timeout: Request timeout
        
    Returns:
        Distributed public key as hex string
        
    Raises:
        ValidationError: If address is invalid
        NetworkError: If request fails
        ResponseError: If response is invalid
    """
    validated_address = validate_address(address)
    
    # Check cache first
    cache_key = _cache_key("distkey", validated_address, tls)
    cached_result = _cache_get(cache_key)
    if cached_result:
        return cached_result
    
    try:
        url = construct_url(
            address=validated_address, 
            endpoint=ENDPOINTS.DISTKEY.value, 
            tls=tls
        )
        
        response_data = await _make_request(url, session, timeout)
        
        if not isinstance(response_data, dict) or 'key' not in response_data:
            raise ResponseError("Invalid distkey response format")
        
        distkey = response_data['key']
        
        # Validate distkey format
        validate_hex_string(distkey, EXPECTED_DISTKEY_LENGTH)
        
        # Cache the result
        _cache_set(cache_key, distkey)
        
        logger.info(f"Retrieved distkey from {validated_address}")
        
```

## Implementation Notes

- Review the generated code carefully before merging
- Run tests to ensure compatibility: `python -m pytest tests/`
- Check code style: `flake8 drand/`
- Verify cryptographic functions work correctly
