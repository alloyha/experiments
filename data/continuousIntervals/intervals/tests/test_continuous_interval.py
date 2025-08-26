import pytest

from src.intervals import Point, ContinuousInterval
from src.utils import ContinuousIntervalError

# Test cases for length()
length_test_token = "interval, expected_length"
length_test_cases = [
    ((0, 5, False, False), 5),
    ((0, 10, True, False), 10),
    ((-5, 5, False, True), 10),
    ((-10, 10, True, True), 20),
    ((0, 0, False, False), 0),  # Zero-length interval
    ((0, 0, True, True), 0),  # Zero-length interval
]

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
        assert str(exc_info.value) == ContinuousIntervalError(operation, other)

@pytest.mark.parametrize(length_test_token, length_test_cases, indirect=["interval"])
def test_continuous_interval_length(interval, expected_length):
    start, end, is_start_open, is_end_open = interval
    interval_obj = ContinuousInterval(start, end, is_start_open, is_end_open)
    assert interval_obj.length() == expected_length

def test_init_valid_interval():
    interval = ContinuousInterval(0, 10)
    assert interval.start == 0
    assert interval.end == 10
    assert interval.is_start_open is False
    assert interval.is_end_open is False

def test_init_invalid_start_greater_than_end():
    with pytest.raises(ValueError) as exc_info:
        ContinuousInterval(10, 0)
    assert str(exc_info.value) == "Invalid interval: start must be less or equal than end"

def test_init_invalid_zero_length_interval():
    with pytest.raises(ValueError) as exc_info:
        ContinuousInterval(5, 5)
    assert str(exc_info.value) == "Invalid interval: open interval with zero length. Only start and end equal 0 is allowed!"

def test_lt_continuous_interval():
    # Interval1: [1, 5)
    interval1 = ContinuousInterval(1, 5, is_start_open=False, is_end_open=True)

    # Interval2: [2, 4)
    interval2 = ContinuousInterval(2, 4, is_start_open=False, is_end_open=True)

    # Interval3: [1, 5)
    interval3 = ContinuousInterval(1, 5, is_start_open=False, is_end_open=True)

    # Interval4: (4, 6]
    interval4 = ContinuousInterval(4, 6, is_start_open=True, is_end_open=False)

    # Interval5: [1, 5)
    interval5 = ContinuousInterval(1, 5, is_start_open=False, is_end_open=True)

    # Interval6: [6, 10)
    interval6 = ContinuousInterval(6, 10, is_start_open=True, is_end_open=True)

    assert not(interval2 < interval1) and not (interval1 < interval2)  # Interval1 is within Interval2
    assert not (interval1 < interval3) and not (interval3 < interval1)  # Interval1 and Interval3 are equal
    assert interval2 < interval4  # Interval2 ends before Interval4 starts
    assert not (interval4 < interval2)  # Interval4 starts after Interval2 ends
    assert not (interval3 < interval5) and not (interval5 < interval3)  # Interval3 and Interval5 are equal
    assert interval4 < interval6  # Interval6 starts after Interval4 ends

    assert interval2 <= interval3  # Interval2 ends before Interval3 starts
    assert interval3 <= interval5  # Interval3 and Interval5 are equal
    assert interval4 <= interval4  # Interval4 is equal to itself

    assert not (interval2 >= interval3)  # Interval2 ends before Interval3 starts
    assert interval3 >= interval5  # Interval3 and Interval5 are equal
    assert interval4 >= interval4  # Interval4 is equal to itself

    assert not (interval2 > interval4)  # Interval2 ends before Interval4 starts
    assert not (interval3 > interval5)  # Interval3 and Interval5 are equal
    assert interval6 > interval4  # Interval6 starts after Interval4 ends

    # Comparing with objects of different types should raise TypeError
    with pytest.raises(TypeError):
        interval1 < "interval"

    with pytest.raises(TypeError):
        interval2 <= 10

    with pytest.raises(TypeError):
        interval3 >= [1, 5]

    with pytest.raises(TypeError):
        interval4 > {"start": 4, "end": 6}

def test_disjoint_intervals():
    interval1 = ContinuousInterval(1, 5)
    interval2 = ContinuousInterval(7, 10)
    assert not interval1.overlaps(interval2)

def test_touching_intervals_with_open_ends():
    interval1 = ContinuousInterval(1, 5, is_end_open=True)
    interval2 = ContinuousInterval(5, 10, is_start_open=True)
    assert not interval1.overlaps(interval2)

'''
def test_touching_intervals_with_closed_ends():
    interval1 = ContinuousInterval(1, 5)
    interval2 = ContinuousInterval(5, 10)
    assert interval1.overlaps(interval2)
'''
    
def test_overlapping_intervals_with_mixed_ends():
    interval1 = ContinuousInterval(1, 5, is_end_open=True)
    interval2 = ContinuousInterval(4, 8, is_start_open=True)
    assert interval1.overlaps(interval2)

# Test cases for contains()
contains_test_token = "interval, item, expected_result"
contains_test_cases = [
    # Interval containment
    ((0, 5, False, False), (1, 4, False, False), True),
    ((0, 5, False, False), (0, 5, False, False), True),
    ((0, 5, False, False), (0, 5, True, True), False),
    ((0, 5, False, False), (3, 6, False, False), False),
    ((0, 5, False, True), (5, 10, False, False), False),
    ((0, 5, False, True), (0, 5, False, False), False),
    ((0, 5, False, True), (0, 5, True, True), False),
    ((0, 5, True, True), (0, 5, False, False), False),

    # Point containment
    ((0, 5, False, False), 2, True),
    ((0, 5, False, False), 0, True),
    ((0, 5, True, False), 0, False),
    ((0, 5, True, False), 5, True),
    ((0, 5, False, True), 5, False),
    ((0, 5, True, True), 0, False),
    ((-5, 5, False, False), -10, False),
    ((-5, 5, False, False), 10, False),
]

@pytest.mark.parametrize(contains_test_token, contains_test_cases)
def test_continuous_interval_contains(interval, item, expected_result):
    interval_obj = ContinuousInterval(*interval)

    if isinstance(item, tuple):
        # Interval containment
        interval_obj2 = ContinuousInterval(*item)
        assert interval_obj.contains(interval_obj2) == expected_result
    else:
        # Point containment
        point_obj = Point(item)
        assert interval_obj.contains(point_obj) == expected_result

is_overlapping_test_token = "interval1, interval2, expected_result"
# Test cases for is_overlapping()
is_overlapping_test_cases = [
    ((0, 5, False, False), (3, 8, False, False), True),
    ((0, 5, False, False), (5, 10, False, False), True),
    ((0, 5, False, False), (5, 10, True, False), False),
    ((0, 5, True, False), (5, 10, True, False), False),
    ((0, 5, False, False), (5, 10, False, True), True),
    ((0, 5, True, False), (5, 10, False, True), True),
    ((0, 5, True, False), (5, 10, True, True), False),
    ((0, 5, True, False), (10, 15, False, False), False),
]

@pytest.mark.parametrize(is_overlapping_test_token, is_overlapping_test_cases, indirect=["interval1", "interval2"])
def test_continuous_interval_is_overlapping(interval1, interval2, expected_result):
    start1, end1, is_start_open1, is_end_open1 = interval1
    start2, end2, is_start_open2, is_end_open2 = interval2
    
    interval_obj1 = ContinuousInterval(start1, end1, is_start_open1, is_end_open1)
    interval_obj2 = ContinuousInterval(start2, end2, is_start_open2, is_end_open2)
    
    print(interval_obj1.end < interval_obj2.start)
    print(interval_obj1.start > interval_obj2.end)
    print(interval_obj1.end == interval_obj2.start and not interval_obj1.is_end_open and not interval_obj2.is_start_open)
    print(interval_obj1.start == interval_obj2.end and not interval_obj1.is_start_open and not interval_obj2.is_end_open)
    
    assert interval_obj1.is_overlapping(interval_obj2) == expected_result

intersection_test_token = "interval1, interval2, expected_result"
intersection_test_cases = [
    # No overlap
    ((0, 5, False, False), (6, 10, False, False), ContinuousInterval.empty()),
    ((0, 5, False, False), (5, 10, True, False), ContinuousInterval.empty()),
    ((0, 5, False, False), (0, 0, True, True), ContinuousInterval.empty()),  # Interval with zero length

    # Overlap resulting in a point
    ((0, 5, False, False), (5, 10, False, False), Point(5)),

    # Overlap resulting in a continuous interval
    ((0, 5, False, False), (2, 8, False, False), ContinuousInterval(2, 5, False, False)),
    ((0, 5, False, False), (2, 8, True, True), ContinuousInterval(2, 5, True, False)),
    ((0, 5, True, True), (2, 8, False, False), ContinuousInterval(2, 5, False, True)),
    ((0, 5, True, True), (2, 8, True, True), ContinuousInterval(2, 5, True, True)),

    # Overlap resulting in an empty interval
    ((0, 5, False, False), (5, 10, True, True), ContinuousInterval.empty()),
    ((0, 5, True, True), (5, 10, False, False), ContinuousInterval.empty()),
    ((0, 5, True, True), (5, 10, True, True), ContinuousInterval.empty()),
    
    # One or both intervals are empty
    ((0, 0, True, True), (1, 2, False, True), ContinuousInterval.empty()),
    ((1, 2, False, True), (0, 0, True, True), ContinuousInterval.empty()),
    ((0, 5, False, False), (0, 0, True, True), ContinuousInterval.empty()),
    ((0, 0, True, True), (0, 5, False, False), ContinuousInterval.empty()),
    
    # One interval is entirely contained within the other
    ((1, 9, False, False), (3, 6, False, False), ContinuousInterval(3, 6, False, False)),
    ((3, 6, False, False), (1, 9, False, False), ContinuousInterval(3, 6, False, False)),
    ((1, 9, False, False), (1, 9, False, False), ContinuousInterval(1, 9, False, False)),
    ((3, 6, False, False), (3, 6, False, False), ContinuousInterval(3, 6, False, False)),

    # Additional cases
    ((0, 5, False, False), (5, 10, True, True), ContinuousInterval.empty()),
    ((0, 5, True, True), (5, 10, False, False), ContinuousInterval.empty()),
    ((0, 5, True, True), (5, 10, True, True), ContinuousInterval.empty()),
    
    # One interval is entirely contained within the other
    ((1, 9, False, False), (1, 9, False, False), ContinuousInterval(1, 9, False, False)),
    ((3, 6, False, False), (3, 6, False, False), ContinuousInterval(3, 6, False, False)),
]

@pytest.mark.parametrize(intersection_test_token, intersection_test_cases, indirect=["interval1", "interval2"])
def test_continuous_interval_intersection(interval1, interval2, expected_result):
    interval_obj1 = ContinuousInterval(*interval1)
    interval_obj2 = ContinuousInterval(*interval2)
    expected_result_obj = (
        Point(*expected_result)
        if isinstance(expected_result, tuple)
        else expected_result
    )

    result = interval_obj1.intersection(interval_obj2)
    
    if isinstance(expected_result_obj, ContinuousInterval):
        assert isinstance(result, ContinuousInterval)
        assert result.start == expected_result_obj.start
        assert result.end == expected_result_obj.end
        assert result.is_start_open == expected_result_obj.is_start_open
        assert result.is_end_open == expected_result_obj.is_end_open
    elif isinstance(expected_result_obj, Point):
        assert isinstance(result, Point)
        assert result.value == expected_result_obj.value
    else:
        assert result is None