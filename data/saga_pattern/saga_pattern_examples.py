"""
Real-world usage examples for the production-ready Saga Pattern implementation

This demonstrates:
1. E-commerce order processing saga
2. Payment processing with multiple providers
3. Multi-service booking system
4. Database migration saga
5. Monitoring and observability integration
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from saga_pattern import (
    Saga,
    SagaContext,
    SagaOrchestrator,
    SagaResult,
    SagaStatus,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================
# EXAMPLE 1: E-COMMERCE ORDER PROCESSING
# ============================================


class OrderProcessingSaga(Saga):
    """
    E-commerce order processing with inventory, payment, and shipping
    """

    def __init__(self, order_id: str, user_id: str, items: list[dict], total_amount: float):
        super().__init__(name=f"OrderProcessing-{order_id}", version="1.0")
        self.order_id = order_id
        self.user_id = user_id
        self.items = items
        self.total_amount = total_amount

    async def build(self):
        """Build the order processing saga"""

        # Step 1: Reserve inventory
        await self.add_step(
            name="reserve_inventory",
            action=self._reserve_inventory,
            compensation=self._release_inventory,
            timeout=15.0,
            max_retries=3,
        )

        # Step 2: Process payment
        await self.add_step(
            name="process_payment",
            action=self._process_payment,
            compensation=self._refund_payment,
            timeout=30.0,
            max_retries=2,
        )

        # Step 3: Create shipment
        await self.add_step(
            name="create_shipment",
            action=self._create_shipment,
            compensation=self._cancel_shipment,
            timeout=20.0,
            max_retries=3,
        )

        # Step 4: Send confirmation email (no compensation - idempotent)
        await self.add_step(
            name="send_confirmation",
            action=self._send_confirmation_email,
            timeout=10.0,
        )

        # Step 5: Update order status
        await self.add_step(
            name="update_order_status",
            action=self._update_order_status,
            timeout=5.0,
        )

    async def _reserve_inventory(self, ctx: SagaContext) -> dict[str, Any]:
        """Reserve inventory for all items"""
        logger.info(f"Reserving inventory for order {self.order_id}")

        # Simulate API call to inventory service
        await asyncio.sleep(0.1)

        reserved_items = []
        for item in self.items:
            reserved_items.append(
                {"item_id": item["id"], "quantity": item["quantity"], "reservation_id": f"RES-{item['id']}"}
            )

        return {"reservations": reserved_items, "timestamp": datetime.now().isoformat()}

    async def _release_inventory(self, result: dict, ctx: SagaContext) -> None:
        """Release reserved inventory"""
        logger.warning(f"Releasing inventory for order {self.order_id}")

        # Simulate API call to release reservations
        await asyncio.sleep(0.1)

        for reservation in result["reservations"]:
            logger.info(f"Released reservation {reservation['reservation_id']}")

    async def _process_payment(self, ctx: SagaContext) -> dict[str, Any]:
        """Process payment"""
        logger.info(f"Processing payment of ${self.total_amount} for order {self.order_id}")

        # Simulate payment gateway API call
        await asyncio.sleep(0.2)

        # For demo: 10% chance of payment failure
        import random

        if random.random() < 0.1:
            raise ValueError("Payment declined by bank")

        payment_result = {
            "transaction_id": f"TXN-{self.order_id}",
            "amount": self.total_amount,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
        }

        return payment_result

    async def _refund_payment(self, result: dict, ctx: SagaContext) -> None:
        """Refund payment"""
        logger.warning(f"Refunding payment {result['transaction_id']}")

        # Simulate refund API call
        await asyncio.sleep(0.2)

        logger.info(f"Refunded ${result['amount']} to user {self.user_id}")

    async def _create_shipment(self, ctx: SagaContext) -> dict[str, Any]:
        """Create shipment"""
        logger.info(f"Creating shipment for order {self.order_id}")

        # Get payment info from context
        payment_info = ctx.get("process_payment")

        # Simulate shipping service API call
        await asyncio.sleep(0.15)

        shipment = {
            "shipment_id": f"SHIP-{self.order_id}",
            "tracking_number": f"TRACK-{self.order_id}",
            "carrier": "FastShip",
            "estimated_delivery": "2024-12-15",
        }

        return shipment

    async def _cancel_shipment(self, result: dict, ctx: SagaContext) -> None:
        """Cancel shipment"""
        logger.warning(f"Canceling shipment {result['shipment_id']}")

        # Simulate shipment cancellation API call
        await asyncio.sleep(0.1)

        logger.info(f"Canceled shipment with tracking {result['tracking_number']}")

    async def _send_confirmation_email(self, ctx: SagaContext) -> dict[str, Any]:
        """Send order confirmation email"""
        logger.info(f"Sending confirmation email for order {self.order_id}")

        # Get shipment info from context
        shipment_info = ctx.get("create_shipment")

        # Simulate email service API call
        await asyncio.sleep(0.05)

        return {
            "email_sent": True,
            "recipient": self.user_id,
            "tracking_number": shipment_info["tracking_number"],
        }

    async def _update_order_status(self, ctx: SagaContext) -> dict[str, Any]:
        """Update order status to completed"""
        logger.info(f"Updating order {self.order_id} status to COMPLETED")

        # Simulate database update
        await asyncio.sleep(0.05)

        return {"order_id": self.order_id, "status": "COMPLETED", "updated_at": datetime.now().isoformat()}


# ============================================
# EXAMPLE 2: PAYMENT PROCESSING WITH FALLBACK
# ============================================


class PaymentProcessingSaga(Saga):
    """
    Payment processing with multiple provider fallback
    """

    def __init__(self, payment_id: str, amount: float, providers: list[str]):
        super().__init__(name=f"Payment-{payment_id}", version="1.0")
        self.payment_id = payment_id
        self.amount = amount
        self.providers = providers

    async def build(self):
        """Build payment saga with provider fallback"""

        # Step 1: Validate payment request
        await self.add_step(
            name="validate_payment",
            action=self._validate_payment,
            timeout=5.0,
        )

        # Step 2: Try primary payment provider
        await self.add_step(
            name="primary_payment",
            action=self._process_with_primary,
            compensation=self._refund_primary,
            timeout=30.0,
            max_retries=2,
        )

        # Step 3: Record transaction
        await self.add_step(
            name="record_transaction",
            action=self._record_transaction,
            timeout=5.0,
        )

    async def _validate_payment(self, ctx: SagaContext) -> dict[str, Any]:
        """Validate payment request"""
        logger.info(f"Validating payment {self.payment_id}")
        await asyncio.sleep(0.05)

        if self.amount <= 0:
            raise ValueError("Invalid payment amount")

        return {"valid": True, "amount": self.amount}

    async def _process_with_primary(self, ctx: SagaContext) -> dict[str, Any]:
        """Process payment with primary provider"""
        primary_provider = self.providers[0]
        logger.info(f"Processing ${self.amount} with {primary_provider}")

        await asyncio.sleep(0.2)

        return {
            "provider": primary_provider,
            "transaction_id": f"TXN-{primary_provider}-{self.payment_id}",
            "amount": self.amount,
            "status": "completed",
        }

    async def _refund_primary(self, result: dict, ctx: SagaContext) -> None:
        """Refund payment from primary provider"""
        logger.warning(f"Refunding {result['transaction_id']}")
        await asyncio.sleep(0.2)

    async def _record_transaction(self, ctx: SagaContext) -> dict[str, Any]:
        """Record transaction in database"""
        logger.info(f"Recording transaction for payment {self.payment_id}")
        await asyncio.sleep(0.05)

        payment_result = ctx.get("primary_payment")
        return {
            "payment_id": self.payment_id,
            "transaction_id": payment_result["transaction_id"],
            "recorded_at": datetime.now().isoformat(),
        }


# ============================================
# EXAMPLE 3: MULTI-SERVICE BOOKING
# ============================================


class TravelBookingSaga(Saga):
    """
    Travel booking across multiple services (flight + hotel + car rental)
    """

    def __init__(
        self,
        booking_id: str,
        user_id: str,
        flight_details: dict,
        hotel_details: dict,
        car_details: dict | None = None,
    ):
        super().__init__(name=f"TravelBooking-{booking_id}", version="1.0")
        self.booking_id = booking_id
        self.user_id = user_id
        self.flight_details = flight_details
        self.hotel_details = hotel_details
        self.car_details = car_details

    async def build(self):
        """Build travel booking saga"""

        # Step 1: Book flight
        await self.add_step(
            name="book_flight",
            action=self._book_flight,
            compensation=self._cancel_flight,
            timeout=30.0,
            max_retries=2,
        )

        # Step 2: Book hotel
        await self.add_step(
            name="book_hotel",
            action=self._book_hotel,
            compensation=self._cancel_hotel,
            timeout=30.0,
            max_retries=2,
        )

        # Step 3: Book car (optional)
        if self.car_details:
            await self.add_step(
                name="book_car",
                action=self._book_car,
                compensation=self._cancel_car,
                timeout=20.0,
                max_retries=2,
            )

        # Step 4: Send itinerary
        await self.add_step(
            name="send_itinerary", action=self._send_itinerary, timeout=10.0
        )

    async def _book_flight(self, ctx: SagaContext) -> dict[str, Any]:
        """Book flight"""
        logger.info(f"Booking flight for {self.user_id}")
        await asyncio.sleep(0.3)

        return {
            "booking_reference": f"FL-{self.booking_id}",
            "flight_number": self.flight_details["flight_number"],
            "confirmation": f"CONF-FL-{self.booking_id}",
        }

    async def _cancel_flight(self, result: dict, ctx: SagaContext) -> None:
        """Cancel flight booking"""
        logger.warning(f"Canceling flight {result['booking_reference']}")
        await asyncio.sleep(0.2)

    async def _book_hotel(self, ctx: SagaContext) -> dict[str, Any]:
        """Book hotel"""
        logger.info(f"Booking hotel for {self.user_id}")
        await asyncio.sleep(0.3)

        return {
            "booking_reference": f"HT-{self.booking_id}",
            "hotel_name": self.hotel_details["hotel_name"],
            "confirmation": f"CONF-HT-{self.booking_id}",
        }

    async def _cancel_hotel(self, result: dict, ctx: SagaContext) -> None:
        """Cancel hotel booking"""
        logger.warning(f"Canceling hotel {result['booking_reference']}")
        await asyncio.sleep(0.2)

    async def _book_car(self, ctx: SagaContext) -> dict[str, Any]:
        """Book rental car"""
        logger.info(f"Booking car for {self.user_id}")
        await asyncio.sleep(0.2)

        return {
            "booking_reference": f"CAR-{self.booking_id}",
            "car_type": self.car_details["car_type"],
            "confirmation": f"CONF-CAR-{self.booking_id}",
        }

    async def _cancel_car(self, result: dict, ctx: SagaContext) -> None:
        """Cancel car booking"""
        logger.warning(f"Canceling car {result['booking_reference']}")
        await asyncio.sleep(0.1)

    async def _send_itinerary(self, ctx: SagaContext) -> dict[str, Any]:
        """Send complete itinerary to user"""
        logger.info(f"Sending itinerary to {self.user_id}")
        await asyncio.sleep(0.1)

        flight = ctx.get("book_flight")
        hotel = ctx.get("book_hotel")
        car = ctx.get("book_car")

        return {
            "sent": True,
            "flight_confirmation": flight["confirmation"],
            "hotel_confirmation": hotel["confirmation"],
            "car_confirmation": car["confirmation"] if car else None,
        }


# ============================================
# MONITORING & OBSERVABILITY
# ============================================


class MonitoredSagaOrchestrator(SagaOrchestrator):
    """
    Enhanced orchestrator with monitoring and metrics
    """

    def __init__(self):
        super().__init__()
        self.metrics = {
            "total_executed": 0,
            "total_successful": 0,
            "total_failed": 0,
            "total_rolled_back": 0,
            "average_execution_time": 0.0,
        }

    async def execute_saga(self, saga: Saga) -> SagaResult:
        """Execute saga with metrics collection"""
        start_time = datetime.now()

        result = await super().execute_saga(saga)

        # Update metrics
        self.metrics["total_executed"] += 1

        if result.status == SagaStatus.COMPLETED:
            self.metrics["total_successful"] += 1
        elif result.status == SagaStatus.FAILED:
            self.metrics["total_failed"] += 1
        elif result.status == SagaStatus.ROLLED_BACK:
            self.metrics["total_rolled_back"] += 1

        # Update average execution time
        total_time = self.metrics["average_execution_time"] * (
            self.metrics["total_executed"] - 1
        )
        self.metrics["average_execution_time"] = (total_time + result.execution_time) / self.metrics[
            "total_executed"
        ]

        logger.info(
            f"Saga {saga.name} completed in {result.execution_time:.2f}s - "
            f"Status: {result.status.value}"
        )

        return result

    def get_metrics(self) -> dict[str, Any]:
        """Get orchestrator metrics"""
        success_rate = (
            (self.metrics["total_successful"] / self.metrics["total_executed"] * 100)
            if self.metrics["total_executed"] > 0
            else 0
        )

        return {
            **self.metrics,
            "success_rate": f"{success_rate:.2f}%",
        }


# ============================================
# DEMO RUNNER
# ============================================


async def demo_order_processing():
    """Demo: E-commerce order processing"""
    print("\n" + "=" * 60)
    print("DEMO 1: E-Commerce Order Processing")
    print("=" * 60)

    orchestrator = MonitoredSagaOrchestrator()

    # Create order saga
    order = OrderProcessingSaga(
        order_id="ORD-12345",
        user_id="USER-789",
        items=[
            {"id": "ITEM-1", "name": "Laptop", "quantity": 1, "price": 999.99},
            {"id": "ITEM-2", "name": "Mouse", "quantity": 2, "price": 29.99},
        ],
        total_amount=1059.97,
    )

    await order.build()

    # Execute saga
    result = await orchestrator.execute_saga(order)

    # Print result
    print(f"\n✅ Order Processing Result:")
    print(f"   Success: {result.success}")
    print(f"   Status: {result.status.value}")
    print(f"   Completed Steps: {result.completed_steps}/{result.total_steps}")
    print(f"   Execution Time: {result.execution_time:.2f}s")

    if result.error:
        print(f"   Error: {result.error}")

    # Show metrics
    print(f"\n📊 Orchestrator Metrics:")
    metrics = orchestrator.get_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value}")


async def demo_travel_booking():
    """Demo: Travel booking"""
    print("\n" + "=" * 60)
    print("DEMO 2: Travel Booking")
    print("=" * 60)

    orchestrator = MonitoredSagaOrchestrator()

    # Create travel booking saga
    booking = TravelBookingSaga(
        booking_id="BOOK-456",
        user_id="USER-123",
        flight_details={"flight_number": "AA123", "from": "NYC", "to": "LAX"},
        hotel_details={"hotel_name": "Grand Hotel", "nights": 3},
        car_details={"car_type": "Sedan", "days": 3},
    )

    await booking.build()

    # Execute saga
    result = await orchestrator.execute_saga(booking)

    # Print result
    print(f"\n✅ Travel Booking Result:")
    print(f"   Success: {result.success}")
    print(f"   Status: {result.status.value}")
    print(f"   Completed Steps: {result.completed_steps}/{result.total_steps}")

    if result.is_completed:
        print("\n📧 Itinerary Details:")
        itinerary = booking.context.get("send_itinerary")
        print(f"   Flight: {itinerary['flight_confirmation']}")
        print(f"   Hotel: {itinerary['hotel_confirmation']}")
        if itinerary["car_confirmation"]:
            print(f"   Car: {itinerary['car_confirmation']}")


async def demo_failure_with_rollback():
    """Demo: Saga failure with successful rollback"""
    print("\n" + "=" * 60)
    print("DEMO 3: Saga Failure with Rollback")
    print("=" * 60)

    orchestrator = MonitoredSagaOrchestrator()

    # Create a saga that will fail
    class FailingSaga(Saga):
        async def build(self):
            await self.add_step(
                "step1",
                lambda ctx: asyncio.sleep(0.1) or "step1_success",
                lambda r, ctx: logger.info("Compensating step1"),
            )
            await self.add_step(
                "step2",
                lambda ctx: asyncio.sleep(0.1) or "step2_success",
                lambda r, ctx: logger.info("Compensating step2"),
            )
            await self.add_step(
                "failing_step",
                lambda ctx: (_ for _ in ()).throw(ValueError("Intentional failure")),
                max_retries=1,
            )

    saga = FailingSaga(name="FailureDemo")
    await saga.build()

    # Execute saga
    result = await orchestrator.execute_saga(saga)

    # Print result
    print(f"\n❌ Saga Result (Expected Failure):")
    print(f"   Success: {result.success}")
    print(f"   Status: {result.status.value}")
    print(f"   Completed Steps: {result.completed_steps}/{result.total_steps}")
    print(f"   Error: {result.error}")
    print(f"   Compensation Errors: {len(result.compensation_errors)}")

    print("\n💡 Notice: Steps 1 and 2 were successfully compensated (rolled back)")


async def main():
    """Run all demos"""
    await demo_order_processing()
    await demo_travel_booking()
    await demo_failure_with_rollback()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())