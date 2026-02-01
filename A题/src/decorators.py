"""
================================================================================
MCM 2026 Problem A: Decorators Module
================================================================================

This module contains utility decorators for the battery modeling framework,
including self-healing error handling.

O-Award Compliance:
    - Self-healing error handling ✓

Author: MCM Team 2026
License: Open for academic use
================================================================================
"""

import logging
from functools import wraps
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def self_healing(max_retries: int = 3, fallback: Optional[Callable] = None):
    """
    Decorator for self-healing execution with automatic retry and fallback.
    
    This decorator implements O-Award compliant error handling by automatically
    retrying failed operations and optionally falling back to alternative methods.
    
    Parameters
    ----------
    max_retries : int
        Maximum number of retry attempts before giving up.
        Default is 3 retries.
    fallback : Callable, optional
        Fallback function to execute if all retries fail.
        Should have the same signature as the decorated function.
        If None, raises the last exception after all retries fail.
        
    Returns
    -------
    Callable
        Decorated function with self-healing capability.
        
    Examples
    --------
    >>> @self_healing(max_retries=3)
    ... def load_data(path):
    ...     return pd.read_csv(path)
    
    >>> @self_healing(max_retries=2, fallback=use_default_data)
    ... def fetch_from_api(url):
    ...     return requests.get(url).json()
        
    Notes
    -----
    - Logs warning messages for each failed attempt
    - Logs info message when using fallback
    - Preserves the original function's metadata via functools.wraps
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    logger.warning(f"[Attempt {attempt+1}/{max_retries}] {func.__name__} failed: {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying {func.__name__}...")
            
            if fallback is not None:
                logger.info(f"Using fallback for {func.__name__}")
                return fallback(*args, **kwargs)
            raise last_error
        return wrapper
    return decorator
