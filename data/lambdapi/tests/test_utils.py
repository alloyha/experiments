from lambda_api.utils import is_fail_status_code, is_success_status_code


def test_is_fail_status_code():
    """
    Test cases for the is_fail_status_code function.
    """
    # Test valid failure status codes (within the range 400-599)
    assert is_fail_status_code(400) is True
    assert is_fail_status_code(404) is True
    assert is_fail_status_code(500) is True
    assert is_fail_status_code(599) is True

    # Test valid non-failure status codes (outside the range 400-599)
    assert is_fail_status_code(200) is False
    assert is_fail_status_code(300) is False
    assert is_fail_status_code(399) is False
    assert is_fail_status_code(600) is False

    # Test edge case: lower bound of the range (400)
    assert is_fail_status_code(400) is True

    # Test edge case: upper bound of the range (599)
    assert is_fail_status_code(599) is True


def test_is_success_status_code():
    """
    Test cases for the is_success_status_code function.
    """
    # Test valid success status codes (within the range 200-299)
    assert is_success_status_code(200) is True
    assert is_success_status_code(204) is True
    assert is_success_status_code(299) is True

    # Test valid non-success status codes (outside the range 200-299)
    assert is_success_status_code(100) is False
    assert is_success_status_code(300) is False
    assert is_success_status_code(400) is False

    # Test edge case: lower bound of the range (200)
    assert is_success_status_code(200) is True

    # Test edge case: upper bound of the range (299)
    assert is_success_status_code(299) is True
