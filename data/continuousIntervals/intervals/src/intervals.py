from .errors import PointError, \
    ContinuousIntervalError

from .utils import continuous_interval_values

class EmptySet:
    pass

class Point:
    def __init__(self, value: float):
        self.value = value

    def __eq__(self, other) -> bool:
        if isinstance(other, Point):
            return self.value == other.value
        else:
            raise PointError('==', other)

    def __ne__(self, other) -> bool:
        if isinstance(other, Point):
            return self.value != other.value
        else:
            raise PointError('!=', other)

    def __lt__(self, other) -> bool:
        if isinstance(other, Point):
            return self.value < other.value
        else:
            raise PointError('<', other)

    def __le__(self, other) -> bool:
        if isinstance(other, Point):
            return self.value <= other.value
        else:
            raise PointError('<=', other)

    def __gt__(self, other) -> bool:
        if isinstance(other, Point):
            return self.value > other.value
        else:
            raise PointError('>', other)

    def __ge__(self, other) -> bool:
        if isinstance(other, Point):
            return self.value >= other.value
        else:
            raise PointError('>=', other)

    def __add__(self, other) -> bool:
        if isinstance(other, Point):
            return Point(self.value + other.value)
        else:
            raise PointError('+', other)

    def __sub__(self, other) -> bool:
        if isinstance(other, Point):
            return Point(self.value - other.value)
        else:
            raise PointError('-', other)

    def __repr__(self) -> str:
        return f"Point({self.value})"

class ContinuousInterval:
    def __init__(self, start, end, is_start_open=False, is_end_open=False):
        if start >= end:
            category='Invalid interval'
            reason='start must be less than end'
            inverted_msg=f"{category}: {reason}"
            raise ValueError(inverted_msg)

        self.start = start
        self.end = end
        self.is_start_open = is_start_open
        self.is_end_open = is_end_open
    
    @staticmethod
    def empty():
        return ContinuousInterval(0, 0, True, True)
    
    def is_empty(self) -> bool:
        are_open=self.is_start_open and self.is_end_open
        are_zero=self.start == self.end and self.start == 0
        
        return are_zero and are_open

    def overlaps(self, other) -> bool:
        if self.start < other.end and self.end > other.start:
            if self.start == other.end:
                if self.is_start_open or other.is_end_open:
                    return False
                return True
            if self.end == other.start:
                if self.is_end_open or other.is_start_open:
                    return False
                return True
            if self.start < other.start < self.end or other.start < self.start < other.end:
                return True
        
        return False

    def __eq__(self, other) -> bool:
        if isinstance(other, EmptySet):
            return False
        
        elif isinstance(other, ContinuousInterval):
            self_props=continuous_interval_values(self)
            other_props=continuous_interval_values(other)

            return self_props == other_props
                   
        else:
            raise ContinuousIntervalError('==', other)

    def __ne__(self, other) -> bool:
        if isinstance(other, ContinuousInterval):
            self_props=continuous_interval_values(self)
            other_props=continuous_interval_values(other)

            return self_props != other_props
        else:
            raise ContinuousIntervalError('!=', other)

    def __lt__(self, other) -> bool:
        if isinstance(other, ContinuousInterval):
            end_lesser_than_start = self.end < other.start
            open_overlapping = self.end == other.start and \
                (self.is_end_open or other.is_start_open)
            
            return end_lesser_than_start or open_overlapping
        else:
            raise ContinuousIntervalError('<', other)

    def __le__(self, other) -> bool:
        raise ContinuousIntervalError('<=', other)
    
    def __gt__(self, other) -> bool:
        if isinstance(other, ContinuousInterval):
            start_greater_than_end = self.start > other.end
            open_overlapping = self.start == other.end and \
                (self.is_start_open or other.is_end_open)
            
            return start_greater_than_end or open_overlapping
        else:
            raise ContinuousIntervalError('>', other)

    def __ge__(self, other) -> bool:
        raise ContinuousIntervalError('>=', other)

    def __add__(self, other):
        if isinstance(other, ContinuousInterval):
            is_overlapping=self.end == other.start
            is_overlap_not_open=not (self.is_end_open and other.is_start_open)
            
            if other.is_empty():
                return self
            elif is_overlapping and is_overlap_not_open:
                return ContinuousInterval(self.start, other.end, self.is_start_open, other.is_end_open)
        else:
            raise ContinuousIntervalError('+', other)

    # todo: implement __sub__ method
    def __sub__(self, other):
        raise ContinuousIntervalError('-', other)
    
    def length(self) -> float:
        return self.end - self.start

    def contains_value(self, value: float) -> bool:
        if(self.is_start_open and self.is_end_open):
            return self.start < value < self.end
        elif(not self.is_start_open and self.is_end_open):
            return self.start <= value < self.end
        elif(self.is_start_open and not self.is_end_open):
            return self.start < value <= self.end
        else: 
            return self.start <= value <= self.end

    def contains_point(self, point):
        return self.contains_value(point.value)

    def contains_interval(self, interval):
        return self.contains_point(interval.start) and \
               self.contains_point(interval.end) 

    def contains(self, item):
        if isinstance(item, ContinuousInterval):
            return self.contains_interval(item)
        elif isinstance(item, Point):
            return self.contains_point(item)
        else:
            category='Invalid item'
            reason="Expected ContinuousInterval or Point"
            raise TypeError(f"{category}: {reason}")

    def is_overlapping(self, interval):
        are_not_disjoint=not (self.end < interval.start or self.start > interval.end)
        endpoints_overlap=(self.end == interval.start and \
                           not self.is_end_open and not interval.is_start_open) or \
                         (self.start == interval.end and \
                          not self.is_start_open and not interval.is_end_open)
        has_intersection=(interval.start < self.start < interval.end) or \
                         (interval.start < self.end < interval.end) or \
                         (self.start < interval.start < self.end) or \
                         (self.start < interval.end < self.end)
        
        is_item_inside=self.contains(interval)
        
        return are_not_disjoint and (endpoints_overlap or has_intersection or is_item_inside)
    
    def intersection(self, interval):
        if not self.is_overlapping(interval) or interval.is_empty():
            return ContinuousInterval.empty()

        equal_endpoints = self.start == interval.start and self.end == interval.end
        equal_boundaries = self.is_start_open == interval.is_start_open and self.is_end_open == interval.is_end_open

        if equal_endpoints and equal_boundaries:
            # The intervals are coincidental
            return self

        if self.contains(interval):
            return interval

        if interval.contains(self):
            return self

        if self.start == interval.end and not self.is_start_open and not interval.is_end_open:
            return Point(self.start)

        if self.end == interval.start and not self.is_end_open and not interval.is_start_open:
            return Point(self.end)

        start = max(self.start, interval.start)
        end = min(self.end, interval.end)

        if start > end:
            return ContinuousInterval.empty()

        is_start_open = (start == self.start and self.is_start_open) or (start == interval.start and interval.is_start_open)
        is_end_open = (end == self.end and self.is_end_open) or (end == interval.end and interval.is_end_open)

        return ContinuousInterval(start, end, is_start_open, is_end_open)


    def union(self, interval):
        if not self.is_overlapping(interval):
            # Return the two disjoint intervals as a list
            return [self, interval]

        # Determine the start value
        if self.start < interval.start:
            start = self.start
            is_start_open = self.is_start_open
        elif self.start > interval.start:
            start = interval.start
            is_start_open = interval.is_start_open
        else:
            start = self.start
            is_start_open = self.is_start_open and interval.is_start_open

        # Determine the end value
        if self.end > interval.end:
            end = self.end
            is_end_open = self.is_end_open
        elif self.end < interval.end:
            end = interval.end
            is_end_open = interval.is_end_open
        else:
            end = self.end
            is_end_open = self.is_end_open and interval.is_end_open

        return ContinuousInterval(start, end, is_start_open, is_end_open)

    def difference(self, interval):
        if not self.is_overlapping(interval):
            return [self]

        if self.start >= interval.start and self.end <= interval.end:
            return []

        result = []

        if self.start < interval.start:
            start = self.start
            end = interval.start
            is_start_open = self.is_start_open
            is_end_open = interval.is_start_open
            result.append(ContinuousInterval(start, end, is_start_open, is_end_open))

        if self.end > interval.end:
            start = interval.end
            end = self.end
            is_start_open = interval.is_end_open
            is_end_open = self.is_end_open
            result.append(ContinuousInterval(start, end, is_start_open, is_end_open))

        return result
    
    def __repr__(self):
        left_bracket = ']' if self.is_start_open else '['
        right_bracket = '[' if self.is_end_open else ']'
        
        return f"{left_bracket}{self.start}, {self.end}{right_bracket}"
    
class DisjointInterval:
    def __init__(self, intervals: list):
        self.intervals = intervals

    # TODO: Create private method to verify if there is overlapping intervals 

    def add_interval(self, interval: ContinuousInterval):
        # Add a new continuous interval to the collection
        # FIXME: Verify if it overlaps any existent interval and fix it
        self.intervals.append(interval)

    def merge_overlapping_intervals(self):
        # Merge overlapping intervals within the collection
        # FIXME: Add guard to verify overlapping
        merged_intervals = []
        sorted_intervals = sorted(self.intervals, key=lambda interval: interval.start)
        
        for interval in sorted_intervals:
            if not merged_intervals or merged_intervals[-1].end < interval.start:
                merged_intervals.append(interval)
            else:
                merged_intervals[-1].end = max(merged_intervals[-1].end, interval.end)
        
        self.intervals = merged_intervals

    def get_non_overlapping_intervals(self):
        # Retrieve a list of non-overlapping intervals
        self.merge_overlapping_intervals()
        return self.intervals

    def get_interval_containing_point(self, point):
        # Find the interval (if any) that contains the given point
        for interval in self.intervals:
            if interval.contains_point(point):
                return interval
        
        return None

class IntervalSet:
    def __init__(self, points, intervals, disjoint_intervals):
        self.points = points
        self.intervals = intervals
        self.disjoint_intervals = disjoint_intervals

    def find_intervals_containing_points(self, points):
        # Perform operations involving points and intervals together
        pass

    def merge_overlapping_intervals_within_disjoint_intervals(self):
        # Perform operations involving intervals and disjoint intervals together
        pass