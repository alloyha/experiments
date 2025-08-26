def operandErrorMessage(this_type, operation, other_type):
    classes_msg=f"'{this_type}' and '{other_type}'"
    return f"Unsupported operand type(s) for {operation}: {classes_msg}"

def PointError(operator: str, other: object) -> TypeError:
    error_msg=operandErrorMessage('Point', operator, type(other).__name__)
    return TypeError(error_msg)

def ContinuousIntervalError(operator: str, other: object) -> TypeError:
    error_msg=operandErrorMessage('ContinuousInterval', operator, type(other).__name__)
    return TypeError(error_msg)

def continuous_interval_values(interval) -> tuple:
    return (\
        interval.start, \
        interval.end, \
        interval.is_start_open, \
        interval.is_end_open, \
    )