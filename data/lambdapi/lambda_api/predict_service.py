"""
AWS Lambda Prediction Service

This module defines an AWS Lambda-based prediction service for making predictions with a machine learning model.
It includes functions for handling API requests, validating input data, and formatting responses.

Functions:
    - make_prediction(payload: dict) -> dict: Wrapper function for the model prediction.
    - api_return(body: dict, status: int, error: str = '') -> dict: Create a response in JSON-like format.
    - validate_body(body: Union[str, dict]) -> Tuple[bool, List[Union[int, float, str]]]: Validate input data.
    - validate_event(event: dict, context: dict) -> dict: Validate the request event, including its body.
    - predict(event: dict, context: dict) -> dict: Handle prediction requests and return responses.

Imports:
    - json.loads and json.dumps from the json module
    - logging module for configuring logging settings
    - ALLOWED_TYPES, model_prediction_map, and validate_body functions from model_resolver module
    - are_types function from utils module

Author: Bruno Peixoto
Date: 15 09 2023
"""

from json import loads, dumps, JSONDecodeError
import logging
from typing import Union, List, Tuple

from .model_resolver import ALLOWED_TYPES, model_prediction_map
from .default_values import DEFAULT_TYPE_ERROR_MESSAGE, \
    CLIENT_ERROR_STATUS_CODE, SUCCESS_STATUS_CODE, SERVER_ERROR_STATUS_CODE
from .utils import are_types, is_success_status_code

# Function alias for prediction function wrapping


def make_prediction(payload: dict) -> dict:
    """
    Wrapper function for the model prediction function.

    Args:
        payload (dict): The input data for making predictions.

    Returns:
        dict: The prediction result.
    """
    return model_prediction_map(payload)

# Return JSON-like format for prediction response
def api_return(body: dict, status: int, error: str = "") -> dict:
    """
    Create a JSON-like response format for API responses.

    Args:
        body (dict): The response body data.
        status (int): The HTTP status code.
        error (str): The error message (if any).

    Returns:
        dict: The formatted response.
    """
    response = {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": dumps(body, default=str),
        "isBase64Encoded": False,
    }

    if len(error) != 0:
        response["error_message"] = error

    return response


def list_check(candidate: list) -> Tuple[bool, List[Union[int, float, str]]]:
    # Initialization
    is_valid = True
    payload = []

    are_allowed_types = are_types(candidate, ALLOWED_TYPES)

    if are_allowed_types:
        payload = candidate
    else:
        is_valid = False

    return is_valid, payload


def validate_data(data):
    """
    Validate the data based on the allowed types.

    Args:
        data: The data to validate.
        allowed_types (tuple): Tuple of allowed types.

    Returns:
        Tuple[bool, List[Union[int, float]]]: A tuple containing a boolean indicating validity and a list of valid entries.
    """
    is_valid = True
    payload = []

    if isinstance(data, list):
        is_valid, payload = list_check(data)
    elif isinstance(data, ALLOWED_TYPES):
        payload = [data]
    else:
        is_valid = False

    return is_valid, payload


def validate_body(body: Union[str, list, dict]
                  ) -> Tuple[bool, List[Union[int, float, str]]]:
    """
    Validate the format and types of the input body data.

    Args:
        body (Union[str, list, dict]): The input body data, which can be a :
            1. JSON string;
            2. list;
            3. dictionary.

    Returns:
        Tuple[bool, List[Union[int, float, str]]]: A tuple containing a boolean indicating validity and a list of valid entries.
    """
    # Initialization
    is_valid = True
    payload = []

    try:
        body = loads(body) if isinstance(body, str) else body
    except JSONDecodeError:
        is_valid = False

    # List with valid typed entries
    if isinstance(body, dict):
        # Check if 'data' key exists in the dictionary
        if "data" in body:
            data = body["data"]

            is_valid, payload = validate_data(data)

        else:
            # 'data' key is missing, consider it invalid
            is_valid = False

    #
    elif isinstance(body, list):
        is_valid, payload = list_check(body)

    # Variable type within valid types
    elif isinstance(body, ALLOWED_TYPES):
        payload = [body]

    else:
        is_valid = False

    return is_valid, payload

# Request event validation (i.e. body)
def validate_event(event: dict, context: dict) -> dict:
    """
    Validate the request event, including its body.

    Args:
        event (dict): The request event data.
        context (dict): The Lambda context data.

    Returns:
        dict: The formatted response with validation results.
    """
    # Initialization
    error_msg = ""

    # Validate provided body
    body = event["body"]
    is_valid, payload_list = validate_body(body)

    # Validation step
    if is_valid:
        code = SUCCESS_STATUS_CODE
    else:
        error_msg = DEFAULT_TYPE_ERROR_MESSAGE
        code = CLIENT_ERROR_STATUS_CODE

    return api_return(payload_list, code, error_msg)

# Prediction main map
def predict(event: dict, context: dict) -> dict:
    """
    Handle prediction requests and return predictions along with appropriate HTTP status codes.

    Args:
        event (dict): The request event data.
        context (dict): The Lambda context data.

    Returns:
        dict: The formatted response with prediction results.
    """
    # Initialization
    error_msg = ""
    prediction_result = []
    response = {}

    # Payload validation
    payload = validate_event(event, context)
    status_code = payload["statusCode"]

    is_success = is_success_status_code(status_code)

    if (is_success):
        # Try-catch pattern for consistent handling
        try:
            # Response status code
            status_code = SUCCESS_STATUS_CODE

            # Prediction
            payload_list=loads(payload['body'])
            prediction_result = make_prediction(payload_list)

            # Succeful prediction response
            response = api_return(prediction_result, status_code)

            # Log successful event
            log_message = f"Successful prediction: {prediction_result}"
            logging.info(log_message)

        except Exception as e:
            # Response error
            error_msg = str(e)

            # Response status code
            status_code = SERVER_ERROR_STATUS_CODE

            # Captured error message
            response = api_return(prediction_result, status_code, error_msg)

            # Log unsuccessful event
            log_message = f"Unsuccessful prediction: {error_msg}"
            logging.error(log_message)

    else:
        response = payload

    return response
