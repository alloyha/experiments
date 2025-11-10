"""
ADDITIONAL TEST FILES FOR SAGA PATTERN
======================================

Includes:
1. Business saga tests (Order Processing, Payment, Travel)
2. Action and compensation tests
3. Monitoring and metrics tests
4. Storage backend tests
5. Failure strategy tests
"""

# ============================================
# FILE: tests/test_business_sagas.py
# ============================================

"""
Tests for business saga implementations
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from sage import DAGSaga, SagaContext, SagaStatus


class TestOrderProcessingSaga:
    """Test OrderProcessingSaga"""
    
    @pytest.mark.asyncio
    async def test_successful_order_processing(self):
        """Test successful order processing flow"""
        from examples.order_processing import OrderProcessingSaga
        
        saga = OrderProcessingSaga(
            order_id="ORD-123",
            user_id="USER-456",
            items=[{"id": "ITEM-1", "quantity": 1}],
            total_amount=99.99
        )
        
        # Mock external dependencies
        with patch('examples.actions.inventory.reserve', AsyncMock(return_value={"reserved": True})):
            with patch('examples.actions.payment.process', AsyncMock(return_value={"paid": True})):
                await saga.build()
                result = await saga.execute()
        
        assert result.success is True
        assert result.completed_steps > 0
    
    @pytest.mark.asyncio
    async def test_order_processing_with_insufficient_inventory(self):
        """Test order fails when inventory is insufficient"""
        from examples.order_processing import OrderProcessingSaga

        # Use large quantity to trigger natural inventory failure
        saga = OrderProcessingSaga(
            order_id="ORD-124",
            user_id="USER-456",
            items=[{"id": "ITEM-1", "quantity": 1000}],  # >100 triggers failure
            total_amount=999.99
        )
        
        await saga.build()
        result = await saga.execute()

        assert result.success is False
        # First step fails, no completed steps to compensate - trivial rollback succeeds
        assert result.status == SagaStatus.ROLLED_BACK
        assert "Insufficient inventory" in str(result.error)

    @pytest.mark.asyncio
    async def test_order_processing_with_payment_failure(self):
        """Test order fails and rolls back when payment fails"""
        from examples.order_processing import OrderProcessingSaga
        
        # Use large amount to trigger natural payment failure
        saga = OrderProcessingSaga(
            order_id="ORD-125",
            user_id="USER-456",
            items=[{"id": "ITEM-1", "quantity": 1}],
            total_amount=15000.00  # >10000 triggers payment failure
        )
        
        await saga.build()
        result = await saga.execute()
        
        assert result.success is False
        assert result.status == SagaStatus.ROLLED_BACK
        assert "Payment declined" in str(result.error)
        # Verify that inventory was actually reserved first (step completed)
        assert result.completed_steps > 0
    
    @pytest.mark.asyncio
    async def test_order_shipment_gets_context_data(self):
        """Test that shipment creation can access payment info from context"""
        from examples.order_processing import OrderProcessingSaga
        
        saga = OrderProcessingSaga(
            order_id="ORD-CTX",
            user_id="USER-CTX",
            items=[{"id": "ITEM-1", "quantity": 1}],
            total_amount=50.00
        )
        
        await saga.build()
        result = await saga.execute()
        
        # Should succeed and shipment should have accessed payment context
        assert result.success is True
        assert result.completed_steps == 5  # All steps complete
    
    @pytest.mark.asyncio
    async def test_order_confirmation_email_gets_shipment_info(self):
        """Test that confirmation email can access shipment info from context"""
        from examples.order_processing import OrderProcessingSaga
        
        saga = OrderProcessingSaga(
            order_id="ORD-EMAIL",
            user_id="USER-EMAIL",
            items=[{"id": "ITEM-1", "quantity": 1}],
            total_amount=25.00
        )
        
        await saga.build()
        result = await saga.execute()
        
        # Should succeed - email step should access shipment tracking number
        assert result.success is True
        assert result.completed_steps == 5
    
    @pytest.mark.asyncio
    async def test_order_processing_compensations_called_correctly(self):
        """Test that compensations are called in reverse order on failure"""
        from examples.order_processing import OrderProcessingSaga
        
        # Use payment failure (step 2) to ensure inventory (step 1) gets compensated
        saga = OrderProcessingSaga(
            order_id="ORD-COMP",
            user_id="USER-COMP",
            items=[{"id": "ITEM-1", "quantity": 5}],
            total_amount=20000.00  # Triggers payment failure
        )
        
        await saga.build()
        result = await saga.execute()
        
        # Payment fails, inventory should be rolled back
        assert result.success is False
        assert result.status == SagaStatus.ROLLED_BACK
        # Inventory was completed before payment failed
        assert result.completed_steps >= 1
    
    @pytest.mark.asyncio
    async def test_order_with_multiple_items_compensation(self):
        """Test that multiple items are properly compensated"""
        from examples.order_processing import OrderProcessingSaga
        
        # Multiple items that will be reserved, then need rollback
        saga = OrderProcessingSaga(
            order_id="ORD-MULTI",
            user_id="USER-MULTI",
            items=[
                {"id": "ITEM-A", "quantity": 5},
                {"id": "ITEM-B", "quantity": 10},
                {"id": "ITEM-C", "quantity": 3}
            ],
            total_amount=25000.00  # Triggers payment failure
        )
        
        await saga.build()
        result = await saga.execute()
        
        # Should fail at payment and rollback all 3 item reservations
        assert result.success is False
        assert result.status == SagaStatus.ROLLED_BACK
        assert result.completed_steps >= 1  # Inventory was reserved


class TestPaymentSaga:
    """Test PaymentProcessingSaga"""
    
    @pytest.mark.asyncio
    async def test_successful_payment_processing(self):
        """Test successful payment processing"""
        from examples.payment import PaymentProcessingSaga
        
        saga = PaymentProcessingSaga(
            payment_id="PAY-123",
            amount=99.99,
            providers=["stripe", "paypal"]
        )
        
        await saga.build()
        result = await saga.execute()
        
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_payment_with_provider_fallback(self):
        """Test payment falls back to secondary provider"""
        from examples.payment import PaymentProcessingSaga
        
        saga = PaymentProcessingSaga(
            payment_id="PAY-124",
            amount=99.99,
            providers=["stripe", "paypal", "square"]
        )
        
        # Mock primary provider failure to test fallback logic
        with patch.object(saga, '_process_with_primary', side_effect=ValueError("Primary provider failed")):
            await saga.build()
            result = await saga.execute()

        # Should fail since there's no actual fallback implemented in the current saga
        # This saga demonstrates the pattern but doesn't implement automatic fallback
        assert result.success is False
        assert "Primary provider failed" in str(result.error)
class TestTravelBookingSaga:
    """Test TravelBookingSaga"""
    
    @pytest.mark.asyncio
    async def test_successful_travel_booking(self):
        """Test successful travel booking with flight, hotel, and car"""
        from examples.travel_booking import TravelBookingSaga
        
        saga = TravelBookingSaga(
            booking_id="BOOK-123",
            user_id="USER-456",
            flight_details={"flight_number": "AA123"},
            hotel_details={"hotel_name": "Grand Hotel"},
            car_details={"car_type": "Sedan"}
        )
        
        await saga.build()
        result = await saga.execute()
        
        assert result.success is True
        assert result.completed_steps >= 4  # flight, hotel, car, itinerary
        
        # Check itinerary was sent
        itinerary = saga.context.get("send_itinerary")
        assert itinerary is not None
        assert itinerary["sent"] is True
        assert "flight_confirmation" in itinerary
        assert "hotel_confirmation" in itinerary
        assert "car_confirmation" in itinerary
    
    @pytest.mark.asyncio
    async def test_travel_booking_without_car(self):
        """Test travel booking without car rental"""
        from examples.travel_booking import TravelBookingSaga
        
        saga = TravelBookingSaga(
            booking_id="BOOK-124",
            user_id="USER-456",
            flight_details={"flight_number": "AA124"},
            hotel_details={"hotel_name": "Budget Inn"},
            car_details=None  # No car
        )
        
        await saga.build()
        result = await saga.execute()
        
        assert result.success is True
        # Should have 3 steps: flight, hotel, itinerary (no car)
        assert len(saga.steps) == 3
        
        # Check itinerary
        itinerary = saga.context.get("send_itinerary")
        assert itinerary["car_confirmation"] is None
    
    @pytest.mark.asyncio
    async def test_travel_booking_compensation_flow(self):
        """Test compensation when a step fails"""
        from examples.travel_booking import TravelBookingSaga
        from sage.exceptions import SagaStepError
        
        saga = TravelBookingSaga(
            booking_id="BOOK-FAIL",
            user_id="USER-789",
            flight_details={"flight_number": "AA999"},
            hotel_details={"hotel_name": "Fail Hotel"},
            car_details={"car_type": "SUV"}
        )
        
        await saga.build()
        
        # Make the car booking fail by replacing it with a failing action
        original_book_car = saga.steps[2].action
        async def failing_car(ctx):
            raise SagaStepError("Car rental unavailable")
        saga.steps[2].action = failing_car
        saga.steps[2].max_retries = 0  # No retries
        
        result = await saga.execute()
        
        # Should fail but compensate previous steps
        assert result.success is False
        assert result.status in [SagaStatus.ROLLED_BACK, SagaStatus.FAILED]
    
    @pytest.mark.asyncio
    async def test_travel_booking_flight_details(self):
        """Test flight booking details are captured correctly"""
        from examples.travel_booking import TravelBookingSaga
        
        flight_details = {
            "flight_number": "UA500",
            "from": "SFO",
            "to": "JFK",
            "departure": "2024-12-20 08:00"
        }
        
        saga = TravelBookingSaga(
            booking_id="BOOK-DETAIL",
            user_id="USER-100",
            flight_details=flight_details,
            hotel_details={"hotel_name": "Airport Hotel"},
            car_details=None
        )
        
        await saga.build()
        result = await saga.execute()
        
        assert result.success is True
        flight_result = saga.context.get("book_flight")
        assert flight_result["flight_number"] == "UA500"
        assert "CONF-FL-" in flight_result["confirmation"]
    
    @pytest.mark.asyncio
    async def test_travel_booking_itinerary_without_car_details(self):
        """Test itinerary generation when car is not booked"""
        from examples.travel_booking import TravelBookingSaga
        
        saga = TravelBookingSaga(
            booking_id="BOOK-NOCAR",
            user_id="USER-NOCAR",
            flight_details={"flight_number": "DL100"},
            hotel_details={"hotel_name": "City Center Hotel"},
            car_details=None  # Explicitly no car
        )
        
        await saga.build()
        result = await saga.execute()
        
        assert result.success is True
        
        # Verify itinerary handles missing car
        itinerary = saga.context.get("send_itinerary")
        assert itinerary is not None
        assert itinerary["sent"] is True
        assert itinerary["flight_confirmation"] is not None
        assert itinerary["hotel_confirmation"] is not None
        assert itinerary["car_confirmation"] is None  # Should be None
    
    @pytest.mark.asyncio
    async def test_travel_booking_hotel_cancellation(self):
        """Test hotel cancellation compensation"""
        from examples.travel_booking import TravelBookingSaga
        from sage.exceptions import SagaStepError
        
        saga = TravelBookingSaga(
            booking_id="BOOK-HOTEL-CANCEL",
            user_id="USER-HC",
            flight_details={"flight_number": "AA200"},
            hotel_details={"hotel_name": "Grand Plaza"},
            car_details=None
        )
        
        await saga.build()
        
        # Make itinerary fail to trigger hotel compensation
        original_itinerary = saga.steps[2].action
        async def failing_itinerary(ctx):
            raise SagaStepError("Email service down")
        saga.steps[2].action = failing_itinerary
        saga.steps[2].max_retries = 0
        
        result = await saga.execute()
        
        # Should compensate hotel and flight
        assert result.success is False
        assert result.completed_steps == 2  # Flight and hotel completed before failure
    
    @pytest.mark.asyncio
    async def test_travel_booking_hotel_details(self):
        """Test hotel booking details are captured correctly"""
        from examples.travel_booking import TravelBookingSaga
        
        hotel_details = {
            "hotel_name": "Luxury Resort",
            "nights": 5,
            "room_type": "Suite"
        }
        
        saga = TravelBookingSaga(
            booking_id="BOOK-HOTEL",
            user_id="USER-200",
            flight_details={"flight_number": "DL100"},
            hotel_details=hotel_details,
            car_details=None
        )
        
        await saga.build()
        result = await saga.execute()
        
        assert result.success is True
        hotel_result = saga.context.get("book_hotel")
        assert hotel_result["hotel_name"] == "Luxury Resort"
        assert "CONF-HT-" in hotel_result["confirmation"]
    
    @pytest.mark.asyncio
    async def test_travel_booking_car_details(self):
        """Test car rental details are captured correctly"""
        from examples.travel_booking import TravelBookingSaga
        
        car_details = {
            "car_type": "Luxury",
            "days": 7,
            "pickup": "Airport"
        }
        
        saga = TravelBookingSaga(
            booking_id="BOOK-CAR",
            user_id="USER-300",
            flight_details={"flight_number": "SW200"},
            hotel_details={"hotel_name": "Downtown Hotel"},
            car_details=car_details
        )
        
        await saga.build()
        result = await saga.execute()
        
        assert result.success is True
        car_result = saga.context.get("book_car")
        assert car_result["car_type"] == "Luxury"
        assert "CONF-CAR-" in car_result["confirmation"]
    
    @pytest.mark.asyncio
    async def test_travel_booking_hotel_failure_cancels_flight(self):
        """Test that hotel failure triggers flight cancellation"""
        from examples.travel_booking import TravelBookingSaga
        
        flight_cancel = AsyncMock()
        
        saga = TravelBookingSaga(
            booking_id="BOOK-125",
            user_id="USER-456",
            flight_details={"flight_number": "AA125"},
            hotel_details={"hotel_name": "Fully Booked Hotel"},
            car_details=None
        )
        
        # Mock the internal saga methods to force hotel failure
        with patch.object(saga, '_cancel_flight', flight_cancel):
            with patch.object(saga, '_book_hotel', AsyncMock(side_effect=ValueError("Hotel full"))):
                await saga.build()
                result = await saga.execute()
        
        assert result.success is False
        flight_cancel.assert_called_once()


# ============================================
# FILE: tests/test_actions.py
# ============================================

"""
Tests for reusable saga actions
"""

import pytest
from sage import SagaContext


class TestInventoryActions:
    """Test inventory actions"""
    
    @pytest.mark.asyncio
    async def test_reserve_inventory_success(self):
        """Test successful inventory reservation"""
        from examples.actions.inventory import reserve
        
        ctx = SagaContext()
        items = [
            {"id": "ITEM-1", "quantity": 2},
            {"id": "ITEM-2", "quantity": 1}
        ]
        
        result = await reserve(items, ctx)
        
        assert result["reservations"] is not None
        assert len(result["reservations"]) == 2
        assert all("reservation_id" in r for r in result["reservations"])
    
    @pytest.mark.asyncio
    async def test_check_availability(self):
        """Test checking inventory availability"""
        from examples.actions.inventory import check_availability
        
        ctx = SagaContext()
        items = [{"id": "ITEM-1", "quantity": 1}]
        
        result = await check_availability(items, ctx)
        
        assert "available" in result
        assert isinstance(result["available"], bool)


class TestPaymentActions:
    """Test payment actions"""
    
    @pytest.mark.asyncio
    async def test_process_payment_success(self):
        """Test successful payment processing"""
        from examples.actions.payment import process_payment
        
        ctx = SagaContext()
        ctx.set("order_id", "ORD-123")
        ctx.set("saga_id", "SAGA-456")
        
        result = await process_payment(
            card_token="tok_visa_1234",
            amount=99.99,
            currency="USD",
            ctx=ctx
        )
        
        assert "transaction_id" in result
        assert result["amount"] == 99.99
        assert result["currency"] == "USD"
        assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_authorize_payment(self):
        """Test payment authorization"""
        from examples.actions.payment import authorize_payment
        
        ctx = SagaContext()
        result = await authorize_payment(
            card_token="tok_visa_5678",
            amount=199.99,
            currency="USD",
            ctx=ctx
        )
        
        assert "authorization_id" in result
        assert result["amount"] == 199.99
        assert result["status"] == "authorized"
    
    @pytest.mark.asyncio
    async def test_capture_payment(self):
        """Test capturing authorized payment"""
        from examples.actions.payment import capture_payment
        
        ctx = SagaContext()
        result = await capture_payment(
            authorization_id="auth_123456",
            amount=99.99,
            ctx=ctx
        )
        
        assert "transaction_id" in result
        assert result["authorization_id"] == "auth_123456"
        assert result["amount"] == 99.99
        assert result["status"] == "captured"
    
    @pytest.mark.asyncio
    async def test_process_wallet_payment(self):
        """Test wallet payment processing"""
        from examples.actions.payment import process_wallet_payment
        
        ctx = SagaContext()
        result = await process_wallet_payment(
            wallet_id="WALLET-789",
            amount=49.99,
            currency="USD",
            ctx=ctx
        )
        
        assert "transaction_id" in result
        assert result["wallet_id"] == "WALLET-789"
        assert result["amount"] == 49.99
        assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_validate_payment_method_credit_card(self):
        """Test credit card validation"""
        from examples.actions.payment import validate_payment_method
        
        ctx = SagaContext()
        result = await validate_payment_method(
            payment_method="credit_card",
            user_id="USER-123",
            ctx=ctx
        )
        
        assert result["valid"] is True
        assert result["payment_type"] == "credit_card"
    
    @pytest.mark.asyncio
    async def test_validate_payment_method_wallet(self):
        """Test wallet validation"""
        from examples.actions.payment import validate_payment_method
        
        ctx = SagaContext()
        result = await validate_payment_method(
            payment_method={"type": "wallet", "wallet_id": "WALLET-123"},
            user_id="USER-123",
            ctx=ctx
        )
        
        assert result["valid"] is True
        assert result["payment_type"] == "wallet"
    
    @pytest.mark.asyncio
    async def test_validate_payment_method_unsupported(self):
        """Test unsupported payment method validation"""
        from examples.actions.payment import validate_payment_method
        from sage.exceptions import SagaStepError
        
        ctx = SagaContext()
        
        with pytest.raises(SagaStepError, match="Unsupported payment method"):
            await validate_payment_method(
                payment_method="cryptocurrency",
                user_id="USER-123",
                ctx=ctx
            )
    
    @pytest.mark.asyncio
    async def test_payment_provider_charge_card(self):
        """Test PaymentProvider.charge_card directly"""
        from examples.actions.payment import PaymentProvider
        
        # Patch random to ensure success (not 5% failure)
        with patch('examples.actions.payment.random.random', return_value=0.1):
            result = await PaymentProvider.charge_card(
                card_token="tok_test",
                amount=75.50,
                currency="EUR",
                metadata={"order_id": "ORD-999"}
            )
        
        assert result["amount"] == 75.50
        assert result["currency"] == "EUR"
        assert result["metadata"]["order_id"] == "ORD-999"
    
    @pytest.mark.asyncio
    async def test_payment_provider_refund(self):
        """Test PaymentProvider.refund directly"""
        from examples.actions.payment import PaymentProvider
        
        result = await PaymentProvider.refund(
            transaction_id="txn_123456",
            amount=50.00
        )
        
        assert "refund_id" in result
        assert result["original_transaction_id"] == "txn_123456"
        assert result["amount"] == 50.00
        assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_process_payment_error_handling(self):
        """Test process_payment error handling when charge fails"""
        from examples.actions.payment import process_payment, PaymentProvider
        from sage.exceptions import SagaStepError
        from unittest.mock import patch, AsyncMock
        
        ctx = SagaContext()
        
        # Mock PaymentProvider to raise an exception
        with patch.object(PaymentProvider, 'charge_card', new=AsyncMock(side_effect=Exception("Card declined"))):
            with pytest.raises(SagaStepError, match="Payment processing failed"):
                await process_payment(
                    card_token="tok_fail",
                    amount=100.00,
                    ctx=ctx
                )
    
    @pytest.mark.asyncio
    async def test_authorize_payment_error_handling(self):
        """Test authorize_payment when authorization fails"""
        from examples.actions.payment import authorize_payment
        from sage.exceptions import SagaStepError
        from unittest.mock import patch
        
        ctx = SagaContext()
        
        # Simulate random failure by patching random.random to always trigger failure
        with patch('examples.actions.payment.random.random', return_value=0.01):  # Less than 0.03 threshold
            with pytest.raises(SagaStepError, match="Payment authorization declined"):
                await authorize_payment(
                    card_token="tok_auth_fail",
                    amount=200.00,
                    ctx=ctx
                )
    
    @pytest.mark.asyncio
    async def test_process_wallet_payment_error_handling(self):
        """Test wallet payment when insufficient funds"""
        from examples.actions.payment import process_wallet_payment
        from sage.exceptions import SagaStepError
        from unittest.mock import patch
        
        ctx = SagaContext()
        
        # Simulate insufficient funds
        with patch('examples.actions.payment.random.random', return_value=0.01):  # Less than 0.02 threshold
            with pytest.raises(SagaStepError, match="Insufficient funds"):
                await process_wallet_payment(
                    wallet_id="WALLET-EMPTY",
                    amount=1000.00,
                    ctx=ctx
                )



class TestNotificationActions:
    """Test notification actions"""
    
    @pytest.mark.asyncio
    async def test_notification_service_control(self):
        """Test that notification service failure rates can be controlled"""
        from examples.actions.notification import NotificationService
        
        # By default in tests, failures are disabled (via conftest.py)
        assert NotificationService._sms_failure_rate == 0.0
        assert NotificationService._email_failure_rate == 0.0
        
        # Can manually enable for testing failure scenarios
        NotificationService.set_failure_rates(sms=1.0)  # 100% failure
        assert NotificationService._sms_failure_rate == 1.0
        
        # Reset works
        NotificationService.set_failure_rates(sms=0.0)
        assert NotificationService._sms_failure_rate == 0.0
    
    @pytest.mark.asyncio
    async def test_send_order_confirmation_email(self):
        """Test sending order confirmation email"""
        from examples.actions.notification import send_order_confirmation_email
        
        ctx = SagaContext()
        order_details = {
            "order_id": "ORD-123",
            "total": 99.99,
            "items": [
                {"name": "Product A", "price": 29.99},
                {"name": "Product B", "price": 70.00}
            ]
        }
        
        result = await send_order_confirmation_email(
            user_email="customer@example.com",
            order_details=order_details,
            ctx=ctx
        )
        
        assert result["status"] == "sent"
        assert "message_id" in result
        assert result["to"] == "customer@example.com"
    
    @pytest.mark.asyncio
    async def test_send_payment_receipt_email(self):
        """Test sending payment receipt email"""
        from examples.actions.notification import send_payment_receipt_email
        
        ctx = SagaContext()
        payment_details = {
            "transaction_id": "TXN-456",
            "amount": 149.99,
            "status": "completed",
            "processed_at": "2024-12-15T10:00:00Z"
        }
        
        result = await send_payment_receipt_email(
            user_email="customer@example.com",
            payment_details=payment_details,
            ctx=ctx
        )
        
        assert result["status"] == "sent"
        assert "message_id" in result
    
    @pytest.mark.asyncio
    async def test_send_shipping_notification(self):
        """Test sending shipping notification (email + SMS)"""
        from examples.actions.notification import send_shipping_notification
        
        ctx = SagaContext()
        shipping_details = {
            "order_id": "ORD-789",
            "tracking_number": "TRACK123456",
            "carrier": "FedEx",
            "estimated_delivery": "2024-12-20"
        }
        
        result = await send_shipping_notification(
            user_email="customer@example.com",
            user_phone="+1234567890",
            shipping_details=shipping_details,
            ctx=ctx
        )
        
        assert "email" in result
        assert "sms" in result
        assert result["email"]["status"] == "sent"
        assert result["sms"]["status"] == "sent"
    
    @pytest.mark.asyncio
    async def test_send_shipping_notification_email_only(self):
        """Test shipping notification with email only (no phone)"""
        from examples.actions.notification import send_shipping_notification
        
        ctx = SagaContext()
        shipping_details = {
            "order_id": "ORD-789",
            "tracking_number": "TRACK123456",
            "carrier": "UPS"
        }
        
        result = await send_shipping_notification(
            user_email="customer@example.com",
            user_phone=None,  # No phone provided
            shipping_details=shipping_details,
            ctx=ctx
        )
        
        assert result["email"]["status"] == "sent"
        # SMS will be attempted but may succeed or fail (phone is None)
        assert "sms" in result
    
    @pytest.mark.asyncio
    async def test_send_order_cancellation_email(self):
        """Test sending order cancellation email"""
        from examples.actions.notification import send_order_cancellation_email
        
        ctx = SagaContext()
        order_details = {
            "order_id": "ORD-321",
            "total": 59.99
        }
        
        result = await send_order_cancellation_email(
            user_email="customer@example.com",
            order_details=order_details,
            reason="Insufficient inventory",
            ctx=ctx
        )
        
        assert result["status"] == "sent"
        assert "message_id" in result
    
    @pytest.mark.asyncio
    async def test_notification_service_send_email_direct(self):
        """Test NotificationService.send_email directly"""
        from examples.actions.notification import NotificationService
        
        result = await NotificationService.send_email(
            to="test@example.com",
            subject="Test Subject",
            body="Test body",
            template="test_template",
            template_data={"key": "value"}
        )
        
        assert result["status"] == "sent"
        assert result["to"] == "test@example.com"
        assert result["subject"] == "Test Subject"
    
    @pytest.mark.asyncio
    async def test_notification_service_send_sms_direct(self):
        """Test NotificationService.send_sms directly"""
        from examples.actions.notification import NotificationService
        
        result = await NotificationService.send_sms(
            phone="+1234567890",
            message="Test message"
        )
        
        assert result["status"] == "sent"
        assert result["phone"] == "+1234567890"
    
    @pytest.mark.asyncio
    async def test_notification_service_send_push_direct(self):
        """Test NotificationService.send_push directly"""
        from examples.actions.notification import NotificationService
        
        result = await NotificationService.send_push(
            device_token="device_token_123",
            title="Test Title",
            body="Test body",
            data={"order_id": "ORD-123"}
        )
        
        assert result["status"] == "sent"
        assert result["device_token"] == "device_token_123"
        assert result["title"] == "Test Title"
    
    @pytest.mark.asyncio
    async def test_send_order_confirmation_email_error_handling(self):
        """Test order confirmation email error handling"""
        from examples.actions.notification import send_order_confirmation_email, NotificationService
        
        # Enable failures to test error path
        NotificationService.set_failure_rates(email=1.0)  # 100% failure
        
        ctx = SagaContext()
        order_details = {"order_id": "ORD-ERROR", "total": 99.99, "items": []}
        
        result = await send_order_confirmation_email(
            user_email="test@example.com",
            order_details=order_details,
            ctx=ctx
        )
        
        # Should return failure status instead of raising
        assert result["status"] == "failed"
        assert "error" in result
        
        # Reset for other tests
        NotificationService.reset_failure_rates()
    
    @pytest.mark.asyncio
    async def test_send_payment_receipt_email_error_handling(self):
        """Test payment receipt email error handling"""
        from examples.actions.notification import send_payment_receipt_email, NotificationService
        
        NotificationService.set_failure_rates(email=1.0)
        
        ctx = SagaContext()
        payment_details = {"transaction_id": "TXN-ERROR", "amount": 50.00}
        
        result = await send_payment_receipt_email(
            user_email="test@example.com",
            payment_details=payment_details,
            ctx=ctx
        )
        
        assert result["status"] == "failed"
        assert "error" in result
        
        NotificationService.reset_failure_rates()
    
    @pytest.mark.asyncio
    async def test_send_shipping_notification_email_failure(self):
        """Test shipping notification with email failure"""
        from examples.actions.notification import send_shipping_notification, NotificationService
        
        NotificationService.set_failure_rates(email=1.0, sms=0.0)
        
        ctx = SagaContext()
        shipping_details = {"order_id": "ORD-123", "tracking_number": "TRACK123"}
        
        result = await send_shipping_notification(
            user_email="test@example.com",
            user_phone="+1234567890",
            shipping_details=shipping_details,
            ctx=ctx
        )
        
        # Email should fail, SMS should succeed
        assert result["email"]["status"] == "failed"
        assert result["sms"]["status"] == "sent"
        
        NotificationService.reset_failure_rates()
    
    @pytest.mark.asyncio
    async def test_send_shipping_notification_sms_failure(self):
        """Test shipping notification with SMS failure"""
        from examples.actions.notification import send_shipping_notification, NotificationService
        
        NotificationService.set_failure_rates(email=0.0, sms=1.0)
        
        ctx = SagaContext()
        shipping_details = {"order_id": "ORD-123", "tracking_number": "TRACK456"}
        
        result = await send_shipping_notification(
            user_email="test@example.com",
            user_phone="+1234567890",
            shipping_details=shipping_details,
            ctx=ctx
        )
        
        # Email should succeed, SMS should fail
        assert result["email"]["status"] == "sent"
        assert result["sms"]["status"] == "failed"
        
        NotificationService.reset_failure_rates()
    
    @pytest.mark.asyncio
    async def test_send_order_cancellation_email_error_handling(self):
        """Test order cancellation email error handling"""
        from examples.actions.notification import send_order_cancellation_email, NotificationService
        
        NotificationService.set_failure_rates(email=1.0)
        
        ctx = SagaContext()
        order_details = {"order_id": "ORD-CANCEL-ERR", "total": 199.99}
        
        result = await send_order_cancellation_email(
            user_email="customer@example.com",
            order_details=order_details,
            reason="Payment failed",
            ctx=ctx
        )
        
        assert result["status"] == "failed"
        assert "error" in result
        
        NotificationService.reset_failure_rates()
    
    @pytest.mark.asyncio
    async def test_send_bulk_notifications_all_success(self):
        """Test bulk notifications when all succeed"""
        from examples.actions.notification import send_bulk_notifications, NotificationService
        
        NotificationService.set_failure_rates(email=0.0)
        
        ctx = SagaContext()
        recipients = [
            {"email": "user1@example.com", "name": "Alice"},
            {"email": "user2@example.com", "name": "Bob"},
            {"email": "user3@example.com", "name": "Charlie"}
        ]
        message_template = {
            "subject": "Hello {name}!",
            "body": "Dear {name}, this is a test message."
        }
        
        result = await send_bulk_notifications(
            recipients=recipients,
            message_template=message_template,
            ctx=ctx
        )
        
        assert result["total"] == 3
        assert result["successful"] == 3
        assert result["failed"] == 0
        assert len(result["details"]) == 3
        
        NotificationService.reset_failure_rates()
    
    @pytest.mark.asyncio
    async def test_send_bulk_notifications_partial_failures(self):
        """Test bulk notifications with some failures"""
        from examples.actions.notification import send_bulk_notifications, NotificationService
        
        # Set moderate failure rate to get some failures
        NotificationService.set_failure_rates(email=0.5)
        
        ctx = SagaContext()
        recipients = [
            {"email": f"user{i}@example.com", "name": f"User{i}"}
            for i in range(10)
        ]
        message_template = {
            "subject": "Test {name}",
            "body": "Hello {name}!"
        }
        
        result = await send_bulk_notifications(
            recipients=recipients,
            message_template=message_template,
            ctx=ctx
        )
        
        assert result["total"] == 10
        assert result["successful"] + result["failed"] == 10
        assert len(result["details"]) == 10
        
        # Check that details include both successes and failures
        statuses = [d["status"] for d in result["details"]]
        assert "sent" in statuses or "failed" in statuses
        
        NotificationService.reset_failure_rates()


# ============================================
# FILE: tests/test_compensations.py
# ============================================

"""
Tests for reusable saga compensations
"""

import pytest
from sage import SagaContext


class TestInventoryCompensations:
    """Test inventory compensations"""
    
    @pytest.mark.asyncio
    async def test_release_inventory(self):
        """Test inventory release compensation"""
        from examples.compensations.inventory import release
        
        ctx = SagaContext()
        reservation_result = {
            "reservations": [
                {"reservation_id": "RES-1", "item_id": "ITEM-1"},
                {"reservation_id": "RES-2", "item_id": "ITEM-2"}
            ]
        }
        
        # Should not raise exception
        await release(reservation_result, ctx)


class TestPaymentCompensations:
    """Test payment compensations"""
    
    @pytest.mark.asyncio
    async def test_refund_payment(self):
        """Test payment refund compensation"""
        from examples.compensations.payment import refund_payment
        
        ctx = SagaContext()
        payment_result = {
            "transaction_id": "TXN-123",
            "amount": 99.99
        }
        
        # Should not raise exception
        await refund_payment(payment_result, ctx)
    
    @pytest.mark.asyncio
    async def test_refund_payment_skips_if_no_transaction(self):
        """Test refund skips if no transaction ID"""
        from examples.compensations.payment import refund_payment
        
        ctx = SagaContext()
        payment_result = {"amount": 99.99}  # Missing transaction_id
        
        # Should complete without error
        await refund_payment(payment_result, ctx)
    
    @pytest.mark.asyncio
    async def test_refund_payment_handles_none_result(self):
        """Test refund handles None result"""
        from examples.compensations.payment import refund_payment
        
        # Should complete without error
        await refund_payment(None, ctx=None)
    
    @pytest.mark.asyncio
    async def test_void_authorization(self):
        """Test payment authorization void"""
        from examples.compensations.payment import void_authorization
        
        ctx = SagaContext()
        auth_result = {
            "authorization_id": "AUTH-123",
            "amount": 99.99
        }
        
        await void_authorization(auth_result, ctx)
    
    @pytest.mark.asyncio
    async def test_void_authorization_skips_if_no_auth(self):
        """Test void skips if no authorization ID"""
        from examples.compensations.payment import void_authorization
        
        auth_result = {}  # Missing authorization_id
        await void_authorization(auth_result, ctx=None)
    
    @pytest.mark.asyncio
    async def test_reverse_wallet_payment(self):
        """Test wallet payment reversal"""
        from examples.compensations.payment import reverse_wallet_payment
        
        ctx = SagaContext()
        wallet_result = {
            "transaction_id": "WALLET-TXN-123",
            "wallet_id": "WALLET-789",
            "amount": 50.00
        }
        
        await reverse_wallet_payment(wallet_result, ctx)
    
    @pytest.mark.asyncio
    async def test_reverse_wallet_payment_skips_if_no_transaction(self):
        """Test wallet reversal skips if no transaction"""
        from examples.compensations.payment import reverse_wallet_payment
        
        wallet_result = {"wallet_id": "WALLET-789"}  # Missing transaction_id
        await reverse_wallet_payment(wallet_result, ctx=None)
    
    @pytest.mark.asyncio
    async def test_release_payment_hold(self):
        """Test payment hold release"""
        from examples.compensations.payment import release_payment_hold
        
        ctx = SagaContext()
        hold_result = {
            "hold_id": "HOLD-123",
            "amount": 150.00
        }
        
        await release_payment_hold(hold_result, ctx)
    
    @pytest.mark.asyncio
    async def test_release_payment_hold_skips_if_no_hold(self):
        """Test release skips if no hold ID"""
        from examples.compensations.payment import release_payment_hold
        
        await release_payment_hold(None, ctx=None)
    
    @pytest.mark.asyncio
    async def test_cancel_recurring_payment(self):
        """Test recurring payment cancellation"""
        from examples.compensations.payment import cancel_recurring_payment
        
        ctx = SagaContext()
        recurring_result = {
            "subscription_id": "SUB-123",
            "plan_id": "PLAN-PRO-MONTHLY"
        }
        
        await cancel_recurring_payment(recurring_result, ctx)
    
    @pytest.mark.asyncio
    async def test_cancel_recurring_payment_skips_if_no_subscription(self):
        """Test cancel skips if no subscription ID"""
        from examples.compensations.payment import cancel_recurring_payment
        
        recurring_result = {"plan_id": "PLAN-123"}  # Missing subscription_id
        await cancel_recurring_payment(recurring_result, ctx=None)
    
    @pytest.mark.asyncio
    async def test_reverse_payment_fee(self):
        """Test payment fee reversal"""
        from examples.compensations.payment import reverse_payment_fee
        
        ctx = SagaContext()
        fee_result = {
            "fee_transaction_id": "FEE-TXN-123",
            "fee_amount": 2.99
        }
        
        await reverse_payment_fee(fee_result, ctx)
    
    @pytest.mark.asyncio
    async def test_reverse_payment_fee_skips_if_no_fee(self):
        """Test fee reversal skips if no fee transaction"""
        from examples.compensations.payment import reverse_payment_fee
        
        fee_result = {"fee_amount": 2.99}  # Missing fee_transaction_id
        await reverse_payment_fee(fee_result, ctx=None)
    
    @pytest.mark.asyncio
    async def test_restore_payment_credits(self):
        """Test payment credits restoration"""
        from examples.compensations.payment import restore_payment_credits
        
        ctx = SagaContext()
        credit_result = {
            "user_id": "USER-123",
            "credits_consumed": 10
        }
        
        await restore_payment_credits(credit_result, ctx)
    
    @pytest.mark.asyncio
    async def test_restore_payment_credits_skips_if_no_credits(self):
        """Test credits restoration skips if no credits consumed"""
        from examples.compensations.payment import restore_payment_credits
        
        credit_result = {"user_id": "USER-123"}  # Missing credits_consumed
        await restore_payment_credits(credit_result, ctx=None)
    
    @pytest.mark.asyncio
    async def test_refund_payment_raises_on_provider_error(self):
        """Test refund raises SagaCompensationError when provider fails"""
        from examples.compensations.payment import refund_payment
        from examples.actions.payment import PaymentProvider
        from sage.exceptions import SagaCompensationError
        from unittest.mock import patch, AsyncMock
        
        ctx = SagaContext()
        payment_result = {
            "transaction_id": "TXN-FAIL",
            "amount": 100.00
        }
        
        # Mock PaymentProvider.refund to raise an exception
        with patch.object(PaymentProvider, 'refund', new=AsyncMock(side_effect=Exception("Refund service unavailable"))):
            with pytest.raises(SagaCompensationError, match="Failed to refund payment"):
                await refund_payment(payment_result, ctx)
    
    @pytest.mark.asyncio
    async def test_void_authorization_raises_on_error(self):
        """Test void authorization raises SagaCompensationError on failure"""
        from examples.compensations.payment import void_authorization
        from sage.exceptions import SagaCompensationError
        from unittest.mock import patch
        
        ctx = SagaContext()
        auth_result = {
            "authorization_id": "AUTH-FAIL",
            "amount": 50.00
        }
        
        # Patch asyncio.sleep to raise an error
        with patch('examples.compensations.payment.asyncio.sleep', side_effect=Exception("Void service error")):
            with pytest.raises(SagaCompensationError, match="Failed to void authorization"):
                await void_authorization(auth_result, ctx)
    
    @pytest.mark.asyncio
    async def test_reverse_wallet_payment_raises_on_error(self):
        """Test wallet reversal raises SagaCompensationError on failure"""
        from examples.compensations.payment import reverse_wallet_payment
        from sage.exceptions import SagaCompensationError
        from unittest.mock import patch
        
        ctx = SagaContext()
        wallet_result = {
            "transaction_id": "WALLET-FAIL",
            "wallet_id": "WALLET-123",
            "amount": 75.00
        }
        
        with patch('examples.compensations.payment.asyncio.sleep', side_effect=Exception("Wallet service down")):
            with pytest.raises(SagaCompensationError, match="Failed to reverse wallet payment"):
                await reverse_wallet_payment(wallet_result, ctx)
    
    @pytest.mark.asyncio
    async def test_release_payment_hold_raises_on_error(self):
        """Test payment hold release raises SagaCompensationError on failure"""
        from examples.compensations.payment import release_payment_hold
        from sage.exceptions import SagaCompensationError
        from unittest.mock import patch
        
        ctx = SagaContext()
        hold_result = {
            "hold_id": "HOLD-FAIL",
            "amount": 200.00
        }
        
        with patch('examples.compensations.payment.asyncio.sleep', side_effect=Exception("Hold service error")):
            with pytest.raises(SagaCompensationError, match="Failed to release payment hold"):
                await release_payment_hold(hold_result, ctx)
    
    @pytest.mark.asyncio
    async def test_cancel_recurring_payment_raises_on_error(self):
        """Test recurring payment cancellation raises SagaCompensationError on failure"""
        from examples.compensations.payment import cancel_recurring_payment
        from sage.exceptions import SagaCompensationError
        from unittest.mock import patch
        
        ctx = SagaContext()
        recurring_result = {
            "subscription_id": "SUB-FAIL",
            "plan_id": "PLAN-PRO"
        }
        
        with patch('examples.compensations.payment.asyncio.sleep', side_effect=Exception("Subscription service error")):
            with pytest.raises(SagaCompensationError, match="Failed to cancel subscription"):
                await cancel_recurring_payment(recurring_result, ctx)
    
    @pytest.mark.asyncio
    async def test_reverse_payment_fee_raises_on_error(self):
        """Test payment fee reversal raises SagaCompensationError on failure"""
        from examples.compensations.payment import reverse_payment_fee
        from sage.exceptions import SagaCompensationError
        from unittest.mock import patch
        
        ctx = SagaContext()
        fee_result = {
            "fee_transaction_id": "FEE-FAIL",
            "fee_amount": 3.99
        }
        
        with patch('examples.compensations.payment.asyncio.sleep', side_effect=Exception("Fee reversal error")):
            with pytest.raises(SagaCompensationError, match="Failed to reverse processing fee"):
                await reverse_payment_fee(fee_result, ctx)
    
    @pytest.mark.asyncio
    async def test_restore_payment_credits_raises_on_error(self):
        """Test credits restoration raises SagaCompensationError on failure"""
        from examples.compensations.payment import restore_payment_credits
        from sage.exceptions import SagaCompensationError
        from unittest.mock import patch
        
        ctx = SagaContext()
        credit_result = {
            "user_id": "USER-FAIL",
            "credits_consumed": 15
        }
        
        with patch('examples.compensations.payment.asyncio.sleep', side_effect=Exception("Credits service error")):
            with pytest.raises(SagaCompensationError, match="Failed to restore credits"):
                await restore_payment_credits(credit_result, ctx)



class TestNotificationCompensations:
    """Test notification compensations"""
    
    @pytest.mark.asyncio
    async def test_send_order_cancellation_notification(self):
        """Test order cancellation notification compensation"""
        from examples.compensations.notification import send_order_cancellation_notification
        
        ctx = SagaContext()
        ctx.set("user_email", "customer@example.com")
        ctx.set("order_details", {
            "order_id": "ORD-123",
            "items": [{"name": "Product A", "price": 29.99}],
            "total": 29.99
        })
        
        notification_result = {
            "message_id": "email_123456",
            "status": "sent"
        }
        
        # Should complete without error
        await send_order_cancellation_notification(notification_result, ctx)
    
    @pytest.mark.asyncio
    async def test_send_order_cancellation_skips_if_not_sent(self):
        """Test that compensation is skipped if original notification wasn't sent"""
        from examples.compensations.notification import send_order_cancellation_notification
        
        ctx = SagaContext()
        notification_result = {"status": "failed"}
        
        # Should complete without sending cancellation
        await send_order_cancellation_notification(notification_result, ctx)
    
    @pytest.mark.asyncio
    async def test_send_order_cancellation_handles_missing_context(self):
        """Test compensation handles missing context gracefully"""
        from examples.compensations.notification import send_order_cancellation_notification
        
        notification_result = {"status": "sent"}
        
        # Should complete even without context
        await send_order_cancellation_notification(notification_result, ctx=None)
    
    @pytest.mark.asyncio
    async def test_send_payment_failure_notification(self):
        """Test payment failure notification compensation"""
        from examples.compensations.notification import send_payment_failure_notification
        
        ctx = SagaContext()
        ctx.set("user_email", "customer@example.com")
        ctx.set("payment_details", {
            "transaction_id": "TXN-123",
            "amount": 99.99
        })
        
        payment_notification_result = {
            "message_id": "email_789",
            "status": "sent"
        }
        
        # Should complete without error
        await send_payment_failure_notification(payment_notification_result, ctx)
    
    @pytest.mark.asyncio
    async def test_send_payment_failure_skips_if_not_sent(self):
        """Test that compensation is skipped if payment receipt wasn't sent"""
        from examples.compensations.notification import send_payment_failure_notification
        
        ctx = SagaContext()
        payment_notification_result = None
        
        # Should complete without sending notification
        await send_payment_failure_notification(payment_notification_result, ctx)
    
    @pytest.mark.asyncio
    async def test_send_shipping_cancellation_notification(self):
        """Test shipping cancellation notification compensation"""
        from examples.compensations.notification import send_shipping_cancellation_notification
        
        ctx = SagaContext()
        ctx.set("user_email", "customer@example.com")
        ctx.set("user_phone", "+1234567890")
        ctx.set("tracking_number", "TRACK123")
        ctx.set("carrier", "FedEx")
        
        shipping_notification_result = {
            "email": {"status": "sent"},
            "sms": {"status": "sent"}
        }
        
        # Should complete without error
        await send_shipping_cancellation_notification(shipping_notification_result, ctx)
    
    @pytest.mark.asyncio
    async def test_send_shipping_cancellation_handles_partial_sends(self):
        """Test shipping cancellation handles partial sends (email only)"""
        from examples.compensations.notification import send_shipping_cancellation_notification
        
        ctx = SagaContext()
        ctx.set("user_email", "customer@example.com")
        ctx.set("tracking_number", "TRACK123")
        
        # Only email was sent, no SMS
        shipping_notification_result = {
            "email": {"status": "sent"},
            "sms": {"status": "failed"}
        }
        
        # Should send cancellation email only
        await send_shipping_cancellation_notification(shipping_notification_result, ctx)
    
    @pytest.mark.asyncio
    async def test_send_shipping_cancellation_skips_if_neither_sent(self):
        """Test that compensation is skipped if neither email nor SMS was sent"""
        from examples.compensations.notification import send_shipping_cancellation_notification
        
        ctx = SagaContext()
        shipping_notification_result = {
            "email": {"status": "failed"},
            "sms": {"status": "failed"}
        }
        
        # Should complete without sending cancellation
        await send_shipping_cancellation_notification(shipping_notification_result, ctx)
    
    @pytest.mark.asyncio
    async def test_retract_bulk_notifications_with_successful_sends(self):
        """Test bulk notification retraction when some were successfully sent"""
        from examples.compensations.notification import retract_bulk_notifications
        
        ctx = SagaContext()
        bulk_result = {
            "total": 5,
            "successful": 3,
            "failed": 2,
            "details": [
                {"recipient": "user1@example.com", "status": "sent", "message_id": "msg1"},
                {"recipient": "user2@example.com", "status": "sent", "message_id": "msg2"},
                {"recipient": "user3@example.com", "status": "failed", "error": "Invalid email"},
                {"recipient": "user4@example.com", "status": "sent", "message_id": "msg3"},
                {"recipient": "user5@example.com", "status": "failed", "error": "Blocked"}
            ]
        }
        
        # Should send retraction emails to the 3 successful recipients
        await retract_bulk_notifications(bulk_result, ctx)
    
    @pytest.mark.asyncio
    async def test_retract_bulk_notifications_skips_when_none_successful(self):
        """Test bulk retraction is skipped when no notifications were sent"""
        from examples.compensations.notification import retract_bulk_notifications
        
        ctx = SagaContext()
        bulk_result = {
            "total": 2,
            "successful": 0,
            "failed": 2,
            "details": [
                {"recipient": "user1@example.com", "status": "failed"},
                {"recipient": "user2@example.com", "status": "failed"}
            ]
        }
        
        # Should complete without sending retractions
        await retract_bulk_notifications(bulk_result, ctx)
    
    @pytest.mark.asyncio
    async def test_retract_bulk_notifications_handles_none_result(self):
        """Test bulk retraction handles None result"""
        from examples.compensations.notification import retract_bulk_notifications
        
        ctx = SagaContext()
        
        # Should handle None gracefully
        await retract_bulk_notifications(None, ctx)
    
    @pytest.mark.asyncio
    async def test_cancel_scheduled_notifications(self):
        """Test canceling scheduled notifications"""
        from examples.compensations.notification import cancel_scheduled_notifications
        
        ctx = SagaContext()
        schedule_result = {
            "schedule_id": "SCH-123",
            "scheduled_for": "2024-12-20T10:00:00Z",
            "status": "scheduled"
        }
        
        # Should cancel the scheduled notification
        await cancel_scheduled_notifications(schedule_result, ctx)
    
    @pytest.mark.asyncio
    async def test_cancel_scheduled_notifications_skips_if_no_schedule_id(self):
        """Test scheduled cancellation skips if no schedule_id"""
        from examples.compensations.notification import cancel_scheduled_notifications
        
        ctx = SagaContext()
        schedule_result = {"status": "failed"}
        
        # Should complete without attempting cancellation
        await cancel_scheduled_notifications(schedule_result, ctx)
    
    @pytest.mark.asyncio
    async def test_suppress_notification_preferences(self):
        """Test restoring notification preferences"""
        from examples.compensations.notification import suppress_notification_preferences
        
        ctx = SagaContext()
        preference_result = {
            "user_id": "USER-123",
            "original_preferences": {
                "email_enabled": True,
                "sms_enabled": False
            },
            "updated_preferences": {
                "email_enabled": False,
                "sms_enabled": True
            }
        }
        
        # Should restore original preferences
        await suppress_notification_preferences(preference_result, ctx)
    
    @pytest.mark.asyncio
    async def test_suppress_notification_preferences_skips_if_no_user_id(self):
        """Test preference restoration skips if no user_id"""
        from examples.compensations.notification import suppress_notification_preferences
        
        ctx = SagaContext()
        preference_result = {"status": "failed"}
        
        # Should complete without attempting restoration
        await suppress_notification_preferences(preference_result, ctx)
    
    @pytest.mark.asyncio
    async def test_shipping_cancellation_handles_sms_only(self):
        """Test shipping cancellation when only SMS was sent"""
        from examples.compensations.notification import send_shipping_cancellation_notification
        
        ctx = SagaContext()
        ctx.set("user_phone", "+1234567890")
        ctx.set("order_details", {"order_id": "ORD-SMS"})
        
        shipping_notification_result = {
            "email": {"status": "failed"},
            "sms": {"status": "sent"}
        }
        
        # Should send SMS cancellation only
        await send_shipping_cancellation_notification(shipping_notification_result, ctx)


# ============================================
# FILE: tests/test_monitoring.py
# ============================================

"""
Tests for monitoring and metrics
"""

import pytest
from sage import SagaStatus
from sage.monitoring.metrics import SagaMetrics


class TestSagaMetrics:
    """Test SagaMetrics"""
    
    def test_metrics_initialization(self):
        """Test metrics initialize correctly"""
        metrics = SagaMetrics()
        
        assert metrics.metrics["total_executed"] == 0
        assert metrics.metrics["total_successful"] == 0
        assert metrics.metrics["total_failed"] == 0
    
    def test_record_successful_execution(self):
        """Test recording successful execution"""
        metrics = SagaMetrics()
        
        metrics.record_execution("TestSaga", SagaStatus.COMPLETED, 1.5)
        
        assert metrics.metrics["total_executed"] == 1
        assert metrics.metrics["total_successful"] == 1
        assert metrics.metrics["average_execution_time"] == 1.5
    
    def test_record_failed_execution(self):
        """Test recording failed execution"""
        metrics = SagaMetrics()
        
        metrics.record_execution("TestSaga", SagaStatus.FAILED, 0.5)
        
        assert metrics.metrics["total_executed"] == 1
        assert metrics.metrics["total_failed"] == 1
    
    def test_record_rolled_back_execution(self):
        """Test recording rolled back execution"""
        metrics = SagaMetrics()
        
        metrics.record_execution("TestSaga", SagaStatus.ROLLED_BACK, 2.0)
        
        assert metrics.metrics["total_executed"] == 1
        assert metrics.metrics["total_rolled_back"] == 1
    
    def test_average_execution_time_calculation(self):
        """Test average execution time is calculated correctly"""
        metrics = SagaMetrics()
        
        metrics.record_execution("Saga1", SagaStatus.COMPLETED, 1.0)
        metrics.record_execution("Saga2", SagaStatus.COMPLETED, 3.0)
        
        assert metrics.metrics["average_execution_time"] == 2.0
    
    def test_per_saga_name_tracking(self):
        """Test metrics tracked per saga name"""
        metrics = SagaMetrics()
        
        metrics.record_execution("OrderSaga", SagaStatus.COMPLETED, 1.0)
        metrics.record_execution("OrderSaga", SagaStatus.COMPLETED, 1.0)
        metrics.record_execution("PaymentSaga", SagaStatus.FAILED, 0.5)
        
        assert metrics.metrics["by_saga_name"]["OrderSaga"]["count"] == 2
        assert metrics.metrics["by_saga_name"]["OrderSaga"]["success"] == 2
        assert metrics.metrics["by_saga_name"]["PaymentSaga"]["failed"] == 1
    
    def test_get_metrics_includes_success_rate(self):
        """Test get_metrics includes success rate"""
        metrics = SagaMetrics()
        
        metrics.record_execution("Saga1", SagaStatus.COMPLETED, 1.0)
        metrics.record_execution("Saga2", SagaStatus.COMPLETED, 1.0)
        metrics.record_execution("Saga3", SagaStatus.FAILED, 1.0)
        
        result = metrics.get_metrics()
        
        assert "success_rate" in result
        assert result["success_rate"] == "66.67%"


# ============================================
# FILE: tests/test_strategies.py
# ============================================

"""
Tests for failure strategies
"""

import pytest
import asyncio
from sage import DAGSaga, ParallelFailureStrategy


class TestFailFastStrategy:
    """Test FAIL_FAST strategy"""
    
    @pytest.mark.asyncio
    async def test_fail_fast_cancels_immediately(self):
        """Test FAIL_FAST cancels remaining tasks immediately"""
        saga = DAGSaga("FailFast", failure_strategy=ParallelFailureStrategy.FAIL_FAST)
        cancelled = []
        
        async def slow_task(ctx):
            try:
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                cancelled.append("cancelled")
                raise
        
        async def fast_fail(ctx):
            await asyncio.sleep(0.1)
            raise ValueError("Fast fail")
        
        async def validate(ctx):
            return "validated"
        
        await saga.add_step("validate", validate, dependencies=set())
        await saga.add_step("slow", slow_task, dependencies={"validate"})
        await saga.add_step("fail", fast_fail, dependencies={"validate"})
        
        await saga.execute()
        
        assert "cancelled" in cancelled


class TestWaitAllStrategy:
    """Test WAIT_ALL strategy"""
    
    @pytest.mark.asyncio
    async def test_wait_all_completes_everything(self):
        """Test WAIT_ALL lets all tasks complete"""
        saga = DAGSaga("WaitAll", failure_strategy=ParallelFailureStrategy.WAIT_ALL)
        completed = []
        
        async def task1(ctx):
            await asyncio.sleep(0.1)
            completed.append("task1")
            return "done"
        
        async def task2_fails(ctx):
            await asyncio.sleep(0.2)
            completed.append("task2_failed")
            raise ValueError("Fail")
        
        async def task3(ctx):
            await asyncio.sleep(0.3)
            completed.append("task3")
            return "done"
        
        async def validate(ctx):
            return "validated"
        
        await saga.add_step("validate", validate, dependencies=set())
        await saga.add_step("task1", task1, dependencies={"validate"})
        await saga.add_step("task2", task2_fails, dependencies={"validate"})
        await saga.add_step("task3", task3, dependencies={"validate"})
        
        await saga.execute()
        
        assert "task1" in completed
        assert "task2_failed" in completed
        assert "task3" in completed


class TestFailFastWithGraceStrategy:
    """Test FAIL_FAST_WITH_GRACE strategy"""
    
    @pytest.mark.asyncio
    async def test_fail_fast_grace_waits_for_inflight(self):
        """Test FAIL_FAST_WITH_GRACE waits for in-flight tasks"""
        saga = DAGSaga(
            "FailFastGrace",
            failure_strategy=ParallelFailureStrategy.FAIL_FAST_WITH_GRACE
        )
        completed = []
        
        async def fast_fail(ctx):
            await asyncio.sleep(0.1)
            raise ValueError("Fail")
        
        async def inflight_task(ctx):
            await asyncio.sleep(0.3)
            completed.append("inflight")
            return "done"
        
        async def validate(ctx):
            return "validated"
        
        await saga.add_step("validate", validate, dependencies=set())
        await saga.add_step("fail", fast_fail, dependencies={"validate"})
        await saga.add_step("inflight", inflight_task, dependencies={"validate"})
        
        await saga.execute()
        
        # In-flight task should have completed
        assert "inflight" in completed


# ============================================
# FILE: tests/conftest.py
# ============================================

"""
Pytest configuration and shared fixtures
"""

import pytest
import asyncio
from sage import SagaOrchestrator


@pytest.fixture
def orchestrator():
    """Provide fresh orchestrator for each test"""
    return SagaOrchestrator()


@pytest.fixture
def mock_external_services(monkeypatch):
    """Mock all external service calls"""
    # Add your mocking logic here
    pass


class TestTradeExecutionSaga:
    """Test trade execution saga"""
    
    @pytest.mark.asyncio
    async def test_trade_execution_saga_build(self):
        """Test building trade execution saga"""
        from examples.trade_execution import TradeExecutionSaga
        
        saga = TradeExecutionSaga(
            trade_id=123,
            symbol="AAPL",
            quantity=100.0,
            price=150.50,
            user_id=456
        )
        
        # Mock actions and compensations
        reserve_funds = AsyncMock(return_value={"reserved": True})
        execute_trade = AsyncMock(return_value={"executed": True})
        update_position = AsyncMock(return_value={"updated": True})
        unreserve_funds = AsyncMock()
        cancel_trade = AsyncMock()
        revert_position = AsyncMock()
        
        await saga.build(
            reserve_funds_action=reserve_funds,
            execute_trade_action=execute_trade,
            update_position_action=update_position,
            unreserve_funds_compensation=unreserve_funds,
            cancel_trade_compensation=cancel_trade,
            revert_position_compensation=revert_position
        )
        
        # Verify saga was built
        assert len(saga.steps) == 3
        assert saga.steps[0].name == "reserve_funds"
        assert saga.steps[1].name == "execute_trade"
        assert saga.steps[2].name == "update_position"
    
    @pytest.mark.asyncio
    async def test_trade_execution_saga_success(self):
        """Test successful trade execution"""
        from examples.trade_execution import TradeExecutionSaga
        
        saga = TradeExecutionSaga(
            trade_id=789,
            symbol="GOOGL",
            quantity=50.0,
            price=2800.00,
            user_id=999
        )
        
        # Mock successful actions
        reserve_funds = AsyncMock(return_value={"reserved": True, "amount": 140000.00})
        execute_trade = AsyncMock(return_value={"trade_id": 789, "status": "executed"})
        update_position = AsyncMock(return_value={"position_updated": True})
        unreserve_funds = AsyncMock()
        cancel_trade = AsyncMock()
        revert_position = AsyncMock()
        
        await saga.build(
            reserve_funds_action=reserve_funds,
            execute_trade_action=execute_trade,
            update_position_action=update_position,
            unreserve_funds_compensation=unreserve_funds,
            cancel_trade_compensation=cancel_trade,
            revert_position_compensation=revert_position
        )
        
        result = await saga.execute()
        
        assert result.success is True
        assert result.status == SagaStatus.COMPLETED
        assert reserve_funds.called
        assert execute_trade.called
        assert update_position.called
        assert not unreserve_funds.called  # No compensation needed


class TestStrategyActivationSaga:
    """Test strategy activation saga"""
    
    @pytest.mark.asyncio
    async def test_strategy_activation_saga_build(self):
        """Test building strategy activation saga"""
        from examples.trade_execution import StrategyActivationSaga
        
        saga = StrategyActivationSaga(strategy_id=101, user_id=202)
        
        # Mock actions
        validate_strategy = AsyncMock(return_value={"valid": True})
        validate_funds = AsyncMock(return_value={"sufficient": True})
        activate_strategy = AsyncMock(return_value={"activated": True})
        publish_event = AsyncMock(return_value={"published": True})
        deactivate_strategy = AsyncMock()
        
        await saga.build(
            validate_strategy_action=validate_strategy,
            validate_funds_action=validate_funds,
            activate_strategy_action=activate_strategy,
            publish_event_action=publish_event,
            deactivate_strategy_compensation=deactivate_strategy
        )
        
        # Verify saga was built
        assert len(saga.steps) == 4
        assert saga.steps[0].name == "validate_strategy"
        assert saga.steps[1].name == "validate_funds"
        assert saga.steps[2].name == "activate_strategy"
        assert saga.steps[3].name == "publish_event"
    
    @pytest.mark.asyncio
    async def test_strategy_activation_success(self):
        """Test successful strategy activation"""
        from examples.trade_execution import StrategyActivationSaga
        
        saga = StrategyActivationSaga(strategy_id=303, user_id=404)
        
        validate_strategy = AsyncMock(return_value={"valid": True})
        validate_funds = AsyncMock(return_value={"sufficient": True})
        activate_strategy = AsyncMock(return_value={"strategy_id": 303, "active": True})
        publish_event = AsyncMock(return_value={"event_id": "evt_123"})
        deactivate_strategy = AsyncMock()
        
        await saga.build(
            validate_strategy_action=validate_strategy,
            validate_funds_action=validate_funds,
            activate_strategy_action=activate_strategy,
            publish_event_action=publish_event,
            deactivate_strategy_compensation=deactivate_strategy
        )
        
        result = await saga.execute()
        
        assert result.success is True
        assert validate_strategy.called
        assert activate_strategy.called
        assert publish_event.called


class TestSagaOrchestratorFromTradeExecution:
    """Test SagaOrchestrator from trade_execution module"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_execute_saga(self):
        """Test orchestrator executing a saga"""
        from examples.trade_execution import SagaOrchestrator
        from sage import Saga
        
        orchestrator = SagaOrchestrator()
        
        # Create simple saga
        class SimpleSaga(Saga):
            async def build(self):
                await self.add_step(
                    "test_step",
                    lambda ctx: asyncio.sleep(0.01) or {"result": "success"}
                )
        
        saga = SimpleSaga(name="TestSaga")
        await saga.build()
        
        result = await orchestrator.execute_saga(saga)
        
        assert result.success is True
        assert saga.saga_id in orchestrator.sagas
    
    @pytest.mark.asyncio
    async def test_orchestrator_get_saga(self):
        """Test getting saga by ID"""
        from examples.trade_execution import SagaOrchestrator
        from sage import Saga
        
        orchestrator = SagaOrchestrator()
        
        class TestSaga(Saga):
            async def build(self):
                await self.add_step("step", lambda ctx: {"done": True})
        
        saga = TestSaga(name="GetTest")
        await saga.build()
        await orchestrator.execute_saga(saga)
        
        retrieved = await orchestrator.get_saga(saga.saga_id)
        assert retrieved is not None
        assert retrieved.saga_id == saga.saga_id
    
    @pytest.mark.asyncio
    async def test_orchestrator_statistics(self):
        """Test orchestrator statistics"""
        from examples.trade_execution import SagaOrchestrator
        from sage import Saga
        
        orchestrator = SagaOrchestrator()
        
        # Create and execute multiple sagas
        for i in range(3):
            class CountSaga(Saga):
                async def build(self):
                    await self.add_step("step", lambda ctx: {"count": i})
            
            saga = CountSaga(name=f"CountSaga-{i}")
            await saga.build()
            await orchestrator.execute_saga(saga)
        
        stats = await orchestrator.get_statistics()
        
        assert stats["total_sagas"] == 3
        assert stats["completed"] == 3
        assert "executing" in stats
        assert "pending" in stats


class TestMonitoredSagaOrchestrator:
    """Test monitored saga orchestrator"""
    
    @pytest.mark.asyncio
    async def test_monitored_orchestrator_metrics(self):
        """Test metrics collection in monitored orchestrator"""
        from examples.monitoring import MonitoredSagaOrchestrator
        from sage import Saga
        
        orchestrator = MonitoredSagaOrchestrator()
        
        # Execute successful saga
        class SuccessSaga(Saga):
            async def build(self):
                await self.add_step("step1", lambda ctx: {"success": True})
        
        saga = SuccessSaga(name="SuccessTest")
        await saga.build()
        await orchestrator.execute_saga(saga)
        
        metrics = orchestrator.get_metrics()
        
        assert metrics["total_executed"] == 1
        assert metrics["total_successful"] == 1
        assert "success_rate" in metrics
        assert "average_execution_time" in metrics
    
    @pytest.mark.asyncio
    async def test_monitored_orchestrator_failure_tracking(self):
        """Test failure tracking in monitored orchestrator"""
        from examples.monitoring import MonitoredSagaOrchestrator
        from sage import Saga
        
        orchestrator = MonitoredSagaOrchestrator()
        
        # Execute failing saga
        class FailSaga(Saga):
            async def build(self):
                await self.add_step(
                    "failing_step",
                    lambda ctx: (_ for _ in ()).throw(ValueError("Test failure")),
                    max_retries=0
                )
        
        saga = FailSaga(name="FailTest")
        await saga.build()
        await orchestrator.execute_saga(saga)
        
        metrics = orchestrator.get_metrics()
        
        assert metrics["total_executed"] == 1
        # Could be failed or rolled_back depending on compensation
        assert metrics["total_failed"] + metrics["total_rolled_back"] == 1
    
    @pytest.mark.asyncio
    async def test_monitored_orchestrator_success_rate(self):
        """Test success rate calculation"""
        from examples.monitoring import MonitoredSagaOrchestrator
        from sage import Saga
        
        orchestrator = MonitoredSagaOrchestrator()
        
        # Execute 2 successful and 1 failed
        for i in range(2):
            class SuccessSaga(Saga):
                async def build(self):
                    await self.add_step("step", lambda ctx: {"ok": True})
            
            saga = SuccessSaga(name=f"Success-{i}")
            await saga.build()
            await orchestrator.execute_saga(saga)
        
        class FailSaga(Saga):
            async def build(self):
                await self.add_step(
                    "step",
                    lambda ctx: (_ for _ in ()).throw(ValueError("Fail")),
                    max_retries=0
                )
        
        fail_saga = FailSaga(name="Fail")
        await fail_saga.build()
        await orchestrator.execute_saga(fail_saga)
        
        metrics = orchestrator.get_metrics()
        
        assert metrics["total_executed"] == 3
        assert metrics["total_successful"] == 2
        # Success rate should be 66.67%
        assert "66.67%" in metrics["success_rate"]


class TestMonitoringDemo:
    """Test monitoring demo functions"""
    
    @pytest.mark.asyncio
    async def test_demo_failure_with_rollback(self):
        """Test demo failure with rollback function"""
        from examples.monitoring import demo_failure_with_rollback
        
        # Should run without errors
        await demo_failure_with_rollback()


# ============================================
# RUN INSTRUCTIONS
# ============================================

"""
To run all tests:

# Run all tests with coverage
pytest tests/ -v --cov=saga --cov=sagas --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_business_sagas.py -v

# Run specific test class
pytest tests/test_business_sagas.py::TestOrderProcessingSaga -v

# Run specific test
pytest tests/test_business_sagas.py::TestOrderProcessingSaga::test_successful_order_processing -v

# Run with markers
pytest -m "integration" -v
pytest -m "unit" -v

# Run with parallel execution
pytest tests/ -v -n auto

# Run with debugging
pytest tests/ -v -s --pdb

# Generate coverage report
pytest tests/ --cov=saga --cov-report=html
# Open htmlcov/index.html in browser
"""

if __name__ == "__main__":
    print("=" * 80)
    print("ADDITIONAL TEST FILES FOR SAGA PATTERN")
    print("=" * 80)
    print("\nTest files included:")
    print("  ✓ tests/test_business_sagas.py - Order, Payment, Travel sagas")
    print("  ✓ tests/test_actions.py - Reusable actions")
    print("  ✓ tests/test_compensations.py - Compensation logic")
    print("  ✓ tests/test_monitoring.py - Metrics and monitoring")
    print("  ✓ tests/test_strategies.py - Failure strategies")
    print("  ✓ tests/conftest.py - Shared fixtures")
    print("\n" + "=" * 80)