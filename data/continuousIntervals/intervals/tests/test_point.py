import pytest

from src.intervals import Point, ContinuousInterval
from src.utils import PointError

@pytest.mark.parametrize(
    "operation, other",
    [
        ('==', 'str'),
        ('!=', 10),
        ('<', 'list'),
        ('<=', 4.5),
        ('>', {'key': 'value'}),
        ('>=', [1, 2, 3])
    ]
)
def test_type_error(point, operation, other):
    aliases = {'==': 'eq', '!=': 'ne', '<': 'lt', '<=': 'le', '>': 'gt', '>=': 'ge'}
    op = aliases.get(operation, operation)
    
    with pytest.raises(TypeError) as exc_info:
        getattr(point, f"__{op}__")(other)
        assert str(exc_info.value) == PointError(operation, other)

def test_eq_continuous_interval():
    interval1 = ContinuousInterval(1, 5)
    interval2 = ContinuousInterval(1, 5)
    interval3 = ContinuousInterval(2, 6)
    
    assert interval1 == interval2           # Same start, end, open flags, should be equal
    assert not (interval1 != interval2)     # Should not be not equal
    
    assert not (interval1 == interval3)     # Different start, end, open flags, should not be equal
    assert interval1 != interval3           # Should be not equal
    
    # Comparing with objects of different types should raise TypeError
    with pytest.raises(TypeError):
        interval1 == "interval"
    
    with pytest.raises(TypeError):
        interval1 != 10

def test_point_repr(point):
    assert repr(point) == f"Point({point.value})"

def test_empty_continuous_interval():
    empty_interval = ContinuousInterval.empty()

    assert empty_interval.start == 0
    assert empty_interval.end == 0
    assert empty_interval.is_start_open
    assert empty_interval.is_end_open
    assert empty_interval.is_empty()

@pytest.mark.parametrize("other, expected_result", [
    (Point(5), True),
    (Point(10), False),
])
def test_point_equality(point, other, expected_result):
    assert (point == other) == expected_result
    assert (point != other) == (not expected_result)


@pytest.mark.parametrize("other, expected_result", [
    (Point(10), True),
    (Point(5), False),
    (Point(2), False),
])
def test_point_comparison(point, other, expected_result):
    assert (point < other) == expected_result
    assert (point <= other) == expected_result or (point == other)
    assert (point > other) == (not expected_result) or (point == other)
    assert (point >= other) == (not expected_result) or (point == other)


@pytest.mark.parametrize(
    "other, expected_result", \
    [ (Point(2), Point(7)),]
)
def test_point_arithmetic_addition(point, other, expected_result):
    assert point + other == expected_result


@pytest.mark.parametrize(
    "other, expected_result", \
    [ (Point(2), Point(3)),]
)
def test_point_arithmetic_subtraction(point, other, expected_result):
    assert point - other == expected_result


@pytest.mark.parametrize(
    "invalid_operation", 
    [
        10,      # Comparison with incompatible type
        "test",  # Addition with incompatible type
        "test",  # Subtraction with incompatible type
    ]
)
def test_point_invalid_operations(point, invalid_operation):
    with pytest.raises(TypeError):
        point < invalid_operation

    with pytest.raises(TypeError):
        point + invalid_operation

    with pytest.raises(TypeError):
        point - invalid_operation