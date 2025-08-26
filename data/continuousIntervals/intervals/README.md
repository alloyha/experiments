Continuous and Disjoint Intervals
================================

This project provides functionality to handle continuous and disjoint intervals in Python.


Usage
-----

The project provides a module named `intervals.py` that contains classes and functions for working with continuous and disjoint intervals.

To get started, import the necessary classes and functions in your Python code:

```python
from intervals import ContinuousInterval, DisjointIntervalSet
```

Continuous Intervals
-----

To work with continuous intervals, create instances of the ContinuousInterval class:

```python

# Create a continuous interval [1, 5]
interval = ContinuousInterval(1, 5)

# Access the start and end values
start = interval.start
end = interval.end

# Check if a value is within the interval
is_within = interval.is_within(3)

# Get the length of the interval
length = interval.length()
```

Disjoint Interval Sets
-----

To work with disjoint interval sets, create instances of the DisjointIntervalSet class:

```python

# Create a disjoint interval set
interval_set = DisjointIntervalSet()

# Add intervals to the set
interval_set.add_interval(ContinuousInterval(1, 5))
interval_set.add_interval(ContinuousInterval(8, 10))

# Check if a value is covered by any interval in the set
is_covered = interval_set.is_covered(3)

# Get the union of intervals in the set
union = interval_set.union()

# Get the intersection of intervals in the set
intersection = interval_set.intersection()

# Get the complement of intervals in the set
complement = interval_set.complement()
```

Refer to the module documentation and docstrings for more details on available methods and their usage.

Tests
-----

The project includes a set of tests to ensure the correctness of the interval classes and functions. To run the tests, use the following command:

```shell
pytest
```

Contributing
-----

Contributions to this project are welcome! If you encounter any issues or have suggestions for improvements, please open an issue on the project's GitHub repository.

When contributing, make sure to follow the existing coding style and guidelines. Include tests for any new functionality or bug fixes. Fork the repository, create a new branch for your changes, and submit a pull request with your modifications.

License
-----

This project is licensed under the MIT License.

Acknowledgements
-----

The project was inspired by the need for efficient handling of continuous and disjoint intervals in various applications.

Contact
-----

For any questions or inquiries, feel free to contact the project maintainer at brunolnetto@gmail.com.