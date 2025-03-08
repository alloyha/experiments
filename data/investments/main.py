class Investment:
    def __init__(self, initial_value: float, month_contribution: float, interest_rate: float, duration: int):
        if initial_value < 0 or month_contribution < 0 or duration < 0:
            raise ValueError("Investment values must be non-negative.")
        self.initial_value = initial_value
        self.month_contribution = month_contribution
        self.interest_rate = interest_rate
        self.duration = duration

    def calculate_final_value(self) -> float:
        """Calculates the future value using the compound interest formula."""
        rate = self.interest_rate / 12 / 100

        if rate == 0:  # No interest, simple sum
            return self.initial_value + self.month_contribution * self.duration

        future_value = self.initial_value * (1 + rate) ** self.duration
        future_value += self.month_contribution * ((1 + rate) ** self.duration - 1) / rate

        return future_value

    def calculate_investment_value(self) -> float:
        """Calculates the total amount invested (without interest)."""
        return self.initial_value + self.month_contribution * self.duration

    def calculate_profit_value(self) -> float:
        """Calculates the total profit (final value - invested amount)."""
        return self.calculate_final_value() - self.calculate_investment_value()

    def get_statistics(self) -> dict:
        """Returns investment statistics as a dictionary."""
        return {
            "initial_value": self.initial_value,
            "duration": self.duration,
            "month_contribution": self.month_contribution,
            "interest_rate": self.interest_rate,
            "final_value": self.calculate_final_value(),
            "invested_value": self.calculate_investment_value(),
            "profit": self.calculate_profit_value()
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
        print("-" * 50)

class InvestmentChain:
    def __init__(self, initial_investment: float = 0):
        """Initialize an empty investment chain with an optional initial investment."""
        self.investments = []
        self.initial_investment = initial_investment

    def add_investment(self, month_contribution: float, interest_rate: float, duration: int):
        """Add a new investment period to the chain."""
        initial_value = self.investments[-1].calculate_final_value() if self.investments else self.initial_investment
        self.investments.append(Investment(initial_value, month_contribution, interest_rate, duration))

    def execute(self):
        """Execute all investments in the chain and print results."""
        for i, investment in enumerate(self.investments, start=1):
            investment.print_statistics(i)

    def final_value(self) -> float:
        """Return the total final value after all investments."""
        return self.investments[-1].calculate_final_value() if self.investments else self.initial_investment

    def total_invested(self) -> float:
        """Return the actual total amount invested (excluding reinvested profits)."""
        return self.initial_investment + \
            sum(investment.month_contribution * investment.duration for investment in self.investments)

    def total_profit(self) -> float:
        """Return the total profit from all investments."""
        return self.final_value() - self.total_invested()

    def summary(self):
        """Print a summary of all investments."""
        total_invested = self.total_invested()
        total_profit = self.total_profit()
        profit_percentage = (total_profit / total_invested) * 100 if total_invested > 0 else 0

        print("\n📈 Investment Chain Summary")
        print(f"🔹 Total Invested: $ {total_invested:_.2f}")
        print(f"💰 Total Profit: $ {total_profit:_.2f} ({profit_percentage:.2f}%)")
        print(f"💎 Final Value: $ {self.final_value():_.2f}")
        print("=" * 50)


# Example: Investment Chain
investment_chain = InvestmentChain(initial_investment=100)  # Start with an initial investment

# Invest for 12 months with $500/month
investment_chain.add_investment(month_contribution=100, interest_rate=10, duration=12)

# Continue investing for another 12 months but increase monthly contribution to $800
investment_chain.add_investment(month_contribution=100, interest_rate=11, duration=12)

# Continue investing for 24 more months at $1000/month
investment_chain.add_investment(month_contribution=100, interest_rate=12, duration=12)

# Run the investment chain
investment_chain.execute()

# Print final summary
investment_chain.summary()
