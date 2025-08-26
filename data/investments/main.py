import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Investment:
    def __init__(self, initial_value: float, month_contribution: float, interest_rate: float, duration: int, compounding: int = 12):
        if initial_value < 0 or month_contribution < 0 or duration < 0:
            raise ValueError("Investment values must be non-negative.")
        if interest_rate < 0 or interest_rate > 100:
            raise ValueError("Interest rate should be between 0 and 100%.")

        self.initial_value = initial_value
        self.month_contribution = month_contribution
        self.interest_rate = interest_rate
        self.duration = duration
        self.compounding = compounding
        self.compounding_values = self._calculate_compounding_progression()

    def _calculate_compounding_progression(self):
        """Retorna uma lista com o valor do investimento ao longo dos meses."""
        rate = self.interest_rate / 100 / self.compounding
        values = []
        value = self.initial_value

        for month in range(1, self.duration + 1):
            value *= (1 + rate)
            value += self.month_contribution
            values.append(value)

        return values

    def get_compounding_values(self):
        """Retorna a evolução do investimento ao longo do tempo."""
        return self.compounding_values

    def calculate_investment_value(self) -> float:
        """Calculates the total amount invested (without interest)."""
        return self.initial_value + self.month_contribution * self.duration

    def calculate_final_value(self) -> float:
        """Calculates the future value using the compound interest formula."""
        rate = self.interest_rate / 100 / self.compounding

        # No interest scenario
        if rate == 0:
            return self.calculate_investment_value()

        periods = self.duration / 12 * self.compounding
        future_value = self.initial_value * (1 + rate) ** periods
        future_value += self.month_contribution * ((1 + rate) ** periods - 1) / rate * (12 / self.compounding)

        return future_value

    def calculate_profit_value(self) -> float:
        """Calculates the total profit (final value - invested amount)."""
        return self.calculate_final_value() - self.calculate_investment_value()

    def get_cagr(self) -> float:
        """Calculates Compound Annual Growth Rate (CAGR)."""
        invested_value = self.calculate_investment_value()
        final_value = self.calculate_final_value()
        years = self.duration / 12
        return (final_value / invested_value) ** (1 / years) - 1 if years > 0 else 0

    def get_statistics(self) -> dict:
        invested_value = self.calculate_investment_value()
        final_value = self.calculate_final_value()
        profit = final_value - invested_value
        cagr = self.get_cagr()

        return {
            "initial_value": self.initial_value,
            "duration": self.duration,
            "month_contribution": self.month_contribution,
            "interest_rate": self.interest_rate,
            "compounding": self.compounding,
            "final_value": final_value,
            "invested_value": invested_value,
            "profit": profit,
            "CAGR (%)": cagr * 100
        }


    def print_statistics(self, step: int) -> None:
        """Prints detailed investment statistics for a specific period."""
        stats = self.get_statistics()
        invested_value = stats["invested_value"]
        profit = stats["profit"]
        profit_percentage = (profit / invested_value) * 100 if invested_value > 0 else 0

        print(f"📊 Investment Period {step:2}:")
        print(f"{'Initial Value:':<25} $ {stats['initial_value']:,.2f}")
        print(f"{'Duration (Months):':<25} {stats['duration']:>3}")
        print(f"{'Monthly Contribution:':<25} $ {stats['month_contribution']:,.2f}")
        print(f"{'Interest Rate (%):':<25} {stats['interest_rate']:>5} %")
        print(f"{'Final Value:':<25} $ {stats['final_value']:,.2f}")
        print(f"{'Invested Value:':<25} $ {stats['invested_value']:,.2f}")
        print(f"{'Profit:':<25} $ {stats['profit']:,.2f} ({profit_percentage:,.2f}%)")
        print(f"{'CAGR (Annualized Return):':<25} {stats['CAGR (%)']:,.2f} %")
        print("-" * 50)


class InvestmentChain:
    def __init__(self, initial_investment: float = 0):
        """Initialize an empty investment chain with an optional initial investment."""
        self.investments = []
        self.initial_investment = initial_investment

    def add_investment(self, month_contribution: float, interest_rate: float, duration: int, compounding: int = 12):
        """Add a new investment period to the chain."""
        previous_investment: Investment = self.investments[-1] if self.investments else None
        initial_value = previous_investment.calculate_final_value() if self.investments else self.initial_investment
        current_investment = Investment(initial_value, month_contribution, interest_rate, duration, compounding)
        
        self.investments.append(current_investment)

    def execute(self):
        """Execute all investments in the chain and print results."""
        for i, investment in enumerate(self.investments, start=1):
            investment.print_statistics(i)

    def final_value(self) -> float:
        """Return the total final value after all investments."""
        return self.investments[-1].calculate_final_value() if self.investments else self.initial_investment

    def total_invested(self) -> float:
        """Return the actual total amount invested (excluding reinvested profits)."""
        total_investment = self.initial_investment  # Start with the initial investment

        # Loop through all investments and sum the contributions for each investment
        for inv in self.investments:
            # Add the monthly contributions for the duration of this investment
            total_investment += inv.month_contribution * inv.duration

        return total_investment

    def total_profit(self) -> float:
        """Return the total profit from all investments."""
        return self.final_value() - self.total_invested()

    def summary(self):
        """Print a summary of all investments."""
        total_invested = self.total_invested()
        total_profit = self.total_profit()
        profit_percentage = (total_profit / total_invested) * 100 if total_invested > 0 else 0

        print("\n📈 Investment Chain Summary")
        print(f"🔹 Total Invested: $ {total_invested:,.2f}")
        print(f"💰 Total Profit: $ {total_profit:,.2f} ({profit_percentage:.2f}%)")
        print(f"💎 Final Value: $ {self.final_value():,.2f}")
        print("=" * 50)

    def plot_growth(self):
        """Plot stacked bar chart of investment growth over time (contributions vs. interest)."""
        sns.set_theme(style="whitegrid")  # Use seaborn styling

        months = []
        total_values = []
        contribution_values = []
        interest_values = []
        cumulative_months = 0

        for investment in self.investments:
            monthly_values = investment.get_compounding_values()
            months.extend(range(cumulative_months + 1, cumulative_months + len(monthly_values) + 1))
            total_values.extend(monthly_values)
            cumulative_months += len(monthly_values)

            # Calculate contributions correctly
            total_invested = np.cumsum([investment.month_contribution] * investment.duration) + investment.initial_value
            contribution_values.extend(total_invested)
            interest_values.extend([total - invested for total, invested in zip(monthly_values, total_invested)])

        # Create a stacked bar chart
        plt.figure(figsize=(10, 6))

        # Create the bars for contributions and interest
        plt.bar(months, contribution_values, label="Contributions", color=sns.color_palette("pastel")[0])
        plt.bar(months, interest_values, bottom=contribution_values, label="Interest", color=sns.color_palette("pastel")[1])

        plt.xlabel("Months")
        plt.ylabel("Total Value ($)")
        plt.title("Investment Growth Over Time (Stacked Bar Plot)")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

# Example: Investment Chain
investment_chain = InvestmentChain(initial_investment=0)

# Invest for 12 months with $1000/month at 20% interest
investment_chain.add_investment(month_contribution=1000, interest_rate=12, duration=12)

investment_chain.add_investment(month_contribution=2000, interest_rate=10, duration=12)

investment_chain.add_investment(month_contribution=1000, interest_rate=8, duration=12)

# Run the investment chain
investment_chain.execute()

# Show final summary
investment_chain.summary()

# Plot investment growth
investment_chain.plot_growth()
