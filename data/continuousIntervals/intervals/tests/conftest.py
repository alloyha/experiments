import pytest

from src.intervals import Point, ContinuousInterval
from src.utils import PointError

@pytest.fixture
def point():
    return Point(5)

@pytest.fixture
def interval(request):
    return request.param

@pytest.fixture
def interval1(request):
    return request.param

@pytest.fixture
def interval2(request):
    return request.param