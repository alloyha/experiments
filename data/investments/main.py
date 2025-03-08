def calculate_final_value(\
    initial_value: float, month_contribution: float, interest_rate: float, duration: int\
) -> float:
    """Calculates the future value of an investment using the compound interest formula."""
    rate = interest_rate / 12 / 100  # Convert annual rate to monthly fraction

    if rate == 0:  # No interest, simple sum
        return initial_value + month_contribution * duration

    future_value = initial_value * (1 + rate) ** duration
    future_value += month_contribution * ((1 + rate) ** duration - 1) / rate

    return future_value


def calculate_investment_value(\
    initial_value: float, month_contribution: float, duration: int\
) -> float:
    """Calculates the total amount invested (without interest)."""
    return initial_value + month_contribution * duration


def calculate_profit_value(\
    initial_value: float, month_contribution: float, interest_rate: float, duration: int\
) -> float:
    """Calculates the total profit (final value - invested amount)."""
    future_value = calculate_final_value(initial_value, month_contribution, interest_rate, duration)
    invested_value = calculate_investment_value(initial_value, month_contribution, duration)
    
    return future_value - invested_value

def calculate_values(\
    initial_value: float, month_contribution: float, interest_rate: float, duration: int\
):
    future_value = calculate_final_value(
        initial_value, month_contribution, interest_rate, duration
    )
    invested_value = calculate_investment_value(
        initial_value, month_contribution, duration
    )
    profit_value = calculate_profit_value(
        initial_value, month_contribution, interest_rate, duration
    )

    return future_value, invested_value, profit_value

def print_statistics(initial_value: float, month_contribution: float, interest_rate: float, duration: int) -> None:
    future_value, invested_value, profit_value = calculate_values(\
        initial_value, month_contribution, interest_rate, duration\
    )

    # Output results
    print("Statistics:")

    print(f"Initial value: R$ {initial_value:.2f}")
    print(f"Duration: {duration} meses")
    print(f"Monthly contribution: R$ {month_contribution:.2f}")
    print(f"Yearly interest rate: {interest_rate}% a.a.")

    print(f"End value after {duration} months: $ {future_value:.2f}")
    print(f"Invested value: R$ {invested_value:.2f}")
    print(f"Profit value: $ {profit_value:.2f}")

    print("-----")

# Example usage:
initial_value = 0
month_contribution = 2000/12
duration = 10*12
interest_rate = 12  # Annual interest rate (e.g., 11%)

final_value, _, _ = calculate_values(initial_value, month_contribution, interest_rate, duration)
print_statistics(initial_value, month_contribution, interest_rate, duration)

# Stop monthly investing
duration=45*12

print_statistics(final_value, 0, interest_rate, duration)
