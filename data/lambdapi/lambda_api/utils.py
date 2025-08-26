"""
Module: type_checker

This module provides a function for checking if all elements in a list are of specified types.

Classes:
    None

Functions:
    are_types(candidate: list, types: tuple) -> bool:
        Check if all elements in the 'candidate' list are instances of the types specified in the 'types' tuple.

        Args:
            candidate (list): The list of elements to be checked.
            types (tuple): A tuple of types to check against.

        Returns:
            bool: True if all elements in 'candidate' are instances of the specified types, otherwise False.
"""

from functools import reduce


def are_types(candidate: list, types: tuple) -> bool:
    """
    Check if all elements in the 'candidate' list are instances of the types specified in the 'types' tuple.

    Args:
        candidate (list): The list of elements to be checked.
        types (tuple): A tuple of types to check against.

    Returns:
        bool: True if all elements in 'candidate' are instances of the specified types, otherwise False.
    """
    def is_types_map(is_types_acc, x):
        return is_types_acc and isinstance(x, types)

    return reduce(is_types_map, candidate)


def is_success_status_code(status_code: int) -> bool:
    """
    Check if a given HTTP status code represents a success status code.

    Args:
        status_code (int): The HTTP status code to check.

    Returns:
        bool: True if the status code represents a success (HTTP status codes in the range 200-299),
              False otherwise.
    """
    return status_code >= 200 and status_code < 300


def is_fail_status_code(status_code: int) -> bool:
    """
    Check if a given HTTP status code represents a failure (error) status code.

    Args:
        status_code (int): The HTTP status code to check.

    Returns:
        bool: True if the status code represents a failure (HTTP status codes in the range 400-599),
              False otherwise.
    """
    return status_code >= 400 and status_code < 600
