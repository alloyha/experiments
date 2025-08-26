from unittest.mock import patch

from lambda_api.predict_service import make_prediction, \
    predict, \
    api_return, \
    validate_data, \
    validate_body, \
    validate_event

from lambda_api.default_values import SUCCESS_STATUS_CODE, \
    CLIENT_ERROR_STATUS_CODE, \
    SERVER_ERROR_STATUS_CODE

MOCKED_ALLOWED_TYPES=(int, float)

# Define a mock for model_resolver.py
@patch("lambda_api.predict_service.model_prediction_map")
def test_predict(mock_model_prediction_map):
    # Set up the mock to return a specific result
    # NOTE: Replace with your desired mock result
    mock_model_prediction_map.return_value = [4, 9, 16]

    # Create a sample event for testing
    event = {
        "body": "[1, 3, 4]"
    }

    # Call the predict function with the mock in place
    response = predict(event, {})

    # Assert that the response contains the expected result
    assert response == {
        "statusCode": SUCCESS_STATUS_CODE,
        "headers": {"Content-Type": "application/json"},
        "body": "[4, 9, 16]",
        "isBase64Encoded": False
    }

# Mock functions for testing
def mock_validate_event(event, context):
    return {
        'statusCode': SUCCESS_STATUS_CODE,
        'body': '[1, 2, 3]'  # Sample payload
    }

def mock_make_prediction(payload):
    raise ValueError("Prediction failed")

@patch('lambda_api.predict_service.validate_event', side_effect=mock_validate_event)
@patch('lambda_api.predict_service.make_prediction', side_effect=mock_make_prediction)
def test_predict_error_handling(mock_validate_event, mock_make_prediction):
    # Simulate a request with a successful payload
    event = {}
    context = {}
    response = predict(event, context)

    # Check that the response contains the error message and has a server error status code
    assert response == {
        'statusCode': SERVER_ERROR_STATUS_CODE,
        'headers': {'Content-Type': 'application/json'},
        'body': '[]',  # Empty list for error case
        'isBase64Encoded': False,
        'error_message': 'Prediction failed'  # Expected error message
    }

# Mock functions for testing
def mock_validate_event(event, context):
    # Simulate a request with a client error status code
    return {
        'headers': {'Content-Type': 'application/json'}, \
        'isBase64Encoded': False, \
        'statusCode': CLIENT_ERROR_STATUS_CODE, \
        'body': 'Invalid request'  # Sample error message
    }

@patch('lambda_api.predict_service.validate_event', side_effect=mock_validate_event)
def test_predict_client_error(mock_validate_event):
    # Simulate a request with a client error status code
    event = {}
    context = {}
    response = predict(event, context)

    # Check that the response contains the payload from validate_event
    assert response == {
        'statusCode': CLIENT_ERROR_STATUS_CODE,
        'headers': {'Content-Type': 'application/json'},
        'body': 'Invalid request',  # Expected error message
        'isBase64Encoded': False,
    }

# Test case for make_prediction function
@patch("lambda_api.predict_service.model_prediction_map")
def test_make_prediction(mock_model_prediction_map):
    # Mock the model_prediction_map function

    # Set up the mock to return a specific result
    # Replace with your desired mock result
    mock_model_prediction_map.return_value = [4, 9, 16]

    # Call make_prediction with a sample payload
    result = make_prediction([1, 2, 3])

    # Assert that the result matches the expected output
    assert result == [4, 9, 16]

# Test case for api_return functio
def test_api_return():
    # Call api_return with sample data
    response = api_return({"data": "value"}, SUCCESS_STATUS_CODE, "")

    # Assert the response structure and values
    assert response == {
        "statusCode": SUCCESS_STATUS_CODE,
        "headers": {"Content-Type": "application/json"},
        "body": '{"data": "value"}',
        "isBase64Encoded": False
    }

@patch("lambda_api.predict_service.ALLOWED_TYPES", MOCKED_ALLOWED_TYPES)
def test_validate_data_valid_list():
    # Valid list input
    data = [1, 2, 3]

    is_valid, payload = validate_data(data)

    assert is_valid is True
    assert payload == data

@patch("lambda_api.predict_service.ALLOWED_TYPES", MOCKED_ALLOWED_TYPES)
def test_validate_data_valid_single_value():
    # Valid single value input
    data = 42

    is_valid, payload = validate_data(data)

    assert is_valid is True
    assert payload == [data]

@patch("lambda_api.predict_service.ALLOWED_TYPES", MOCKED_ALLOWED_TYPES)
def test_validate_data_invalid_type():
    # Invalid input type
    data = "invalid"

    is_valid, payload = validate_data(data)

    assert is_valid is False
    assert payload == []

@patch("lambda_api.predict_service.ALLOWED_TYPES", MOCKED_ALLOWED_TYPES)
def test_validate_data_invalid_list():
    # Invalid list input
    data = ["invalid", "data"]

    is_valid, payload = validate_data(data)

    assert is_valid is False
    assert payload == []

@patch("lambda_api.predict_service.ALLOWED_TYPES", MOCKED_ALLOWED_TYPES)
def test_validate_data_custom_list_check():
    # Custom list validation logic
    data = [1, 2, 3]

    with patch("lambda_api.predict_service.list_check", return_value=(False, [])):
        is_valid, payload = validate_data(data)

    assert is_valid is False
    assert payload == []

# Test case for a valid JSON string with 'data' key
def test_validate_body_json():
    body = '{"data": [1, 2, 3]}'
    is_valid, payload = validate_body(body)
    assert is_valid is True
    assert payload == [1, 2, 3]

# Test case for a valid list
def test_validate_body_list():
    body = [1, 2, 3]
    is_valid, payload = validate_body(body)
    assert is_valid is True
    assert payload == [1, 2, 3]

# Test case for a single valid type
def test_validate_body_single_type():
    body = 42
    is_valid, payload = validate_body(body)
    assert is_valid is True
    assert payload == [42]

# Test case for an invalid JSON string
def test_validate_body_invalid_json():
    body = '{"invalid_data": 42}'
    is_valid, payload = validate_body(body)
    assert is_valid is False
    assert payload == []

# Test case for an invalid list with mixed types


def test_validate_body_invalid_list():
    body = [1, "two", 3.0]
    is_valid, payload = validate_body(body)
    assert is_valid is False
    assert payload == []

# Test case for an invalid type (set)
def test_validate_body_invalid_type():
    body = {"key": "value"}
    is_valid, payload = validate_body(body)
    assert is_valid is False
    assert payload == []

# Test case for an empty string
def test_validate_body_empty_string():
    body = ""
    is_valid, payload = validate_body(body)
    assert is_valid is False
    assert payload == []

# Test case for validate_event function
def test_validate_event():
    # Test with valid JSON string in the event body
    event = {"body": '{"data": [1, 2, 3]}'}
    response = validate_event(event, {})
    assert response["statusCode"] == SUCCESS_STATUS_CODE

    # Test with valid list in the event body
    event = {"body": [1, 2, 3]}
    response = validate_event(event, {})
    assert response["statusCode"] == SUCCESS_STATUS_CODE

    # Test with invalid event body
    event = {"body": "invalid_data"}
    response = validate_event(event, {})
    assert response["statusCode"] == CLIENT_ERROR_STATUS_CODE
