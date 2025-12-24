# ============================================
# FILE: saga/monitoring/metrics.py
# ============================================

"""
Metrics collection for sagas
"""

from typing import Any

from sage import SagaStatus


class SagaMetrics:
    """Collect and expose saga metrics"""

    def __init__(self):
        self.metrics = {
            "total_executed": 0,
            "total_successful": 0,
            "total_failed": 0,
            "total_rolled_back": 0,
            "average_execution_time": 0.0,
            "by_saga_name": {},
        }

    def record_execution(self, saga_name: str, status: SagaStatus, duration: float):
        """Record saga execution"""
        self.metrics["total_executed"] += 1

        if status == SagaStatus.COMPLETED:
            self.metrics["total_successful"] += 1
        elif status == SagaStatus.FAILED:
            self.metrics["total_failed"] += 1
        elif status == SagaStatus.ROLLED_BACK:
            self.metrics["total_rolled_back"] += 1

        # Update average
        total_time = self.metrics["average_execution_time"] * (
            self.metrics["total_executed"] - 1
        )
        self.metrics["average_execution_time"] = (
            total_time + duration
        ) / self.metrics["total_executed"]

        # Track per saga name
        if saga_name not in self.metrics["by_saga_name"]:
            self.metrics["by_saga_name"][saga_name] = {
                "count": 0,
                "success": 0,
                "failed": 0,
            }

        self.metrics["by_saga_name"][saga_name]["count"] += 1
        if status == SagaStatus.COMPLETED:
            self.metrics["by_saga_name"][saga_name]["success"] += 1
        else:
            self.metrics["by_saga_name"][saga_name]["failed"] += 1

    def get_metrics(self) -> dict[str, Any]:
        """Get all metrics"""
        success_rate = (
            self.metrics["total_successful"] / self.metrics["total_executed"] * 100
            if self.metrics["total_executed"] > 0
            else 0
        )

        return {
            **self.metrics,
            "success_rate": f"{success_rate:.2f}%",
        }
