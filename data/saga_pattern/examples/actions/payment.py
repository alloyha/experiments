"""
Payment processing actions for saga patterns

Reusable payment actions that can be used across different business sagas.
"""

import asyncio
import random
from typing import Any, Dict
from sage.core import SagaContext
from sage.exceptions import SagaStepError


class PaymentProvider:
    """Mock payment provider for demonstration"""
    
    @staticmethod
    async def charge_card(
        card_token: str, 
        amount: float, 
        currency: str = "USD",
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Simulate charging a credit card"""
        
        await asyncio.sleep(0.1)  # Simulate API latency
        
        # Simulate 5% payment failure rate
        if random.random() < 0.05:
            raise Exception(f"Payment declined for amount ${amount}")
        
        return {
            "transaction_id": f"txn_{random.randint(100000, 999999)}",
            "amount": amount,
            "currency": currency,
            "status": "completed",
            "processed_at": "2024-12-15T10:00:00Z",
            "metadata": metadata or {}
        }
    
    @staticmethod
    async def refund(transaction_id: str, amount: float) -> Dict[str, Any]:
        """Simulate refunding a payment"""
        
        await asyncio.sleep(0.1)
        
        return {
            "refund_id": f"ref_{random.randint(100000, 999999)}",
            "original_transaction_id": transaction_id,
            "amount": amount,
            "status": "completed",
            "refunded_at": "2024-12-15T10:00:00Z",
        }


async def process_payment(
    card_token: str,
    amount: float,
    currency: str = "USD",
    ctx: SagaContext = None
) -> Dict[str, Any]:
    """
    Process a payment transaction
    
    Args:
        card_token: Tokenized card information
        amount: Payment amount
        currency: Currency code
        ctx: Saga context
        
    Returns:
        Payment transaction details
    """
    
    try:
        metadata = {
            "saga_id": ctx.get("saga_id") if ctx else None,
            "order_id": ctx.get("order_id") if ctx else None,
        }
        
        result = await PaymentProvider.charge_card(
            card_token=card_token,
            amount=amount, 
            currency=currency,
            metadata=metadata
        )
        
        return result
        
    except Exception as e:
        raise SagaStepError(f"Payment processing failed: {str(e)}")


async def authorize_payment(
    card_token: str,
    amount: float,
    currency: str = "USD", 
    ctx: SagaContext = None
) -> Dict[str, Any]:
    """
    Authorize a payment without capturing
    
    Args:
        card_token: Tokenized card information
        amount: Authorization amount
        currency: Currency code
        ctx: Saga context
        
    Returns:
        Authorization details
    """
    
    await asyncio.sleep(0.05)  # Simulate API call
    
    # Simulate 3% authorization failure rate
    if random.random() < 0.03:
        raise SagaStepError(f"Payment authorization declined for ${amount}")
    
    return {
        "authorization_id": f"auth_{random.randint(100000, 999999)}",
        "amount": amount,
        "currency": currency,
        "status": "authorized",
        "expires_at": "2024-12-15T11:00:00Z",  # 1 hour expiry
        "authorized_at": "2024-12-15T10:00:00Z",
    }


async def capture_payment(
    authorization_id: str,
    amount: float,
    ctx: SagaContext = None
) -> Dict[str, Any]:
    """
    Capture a previously authorized payment
    
    Args:
        authorization_id: Authorization ID from authorize_payment
        amount: Amount to capture (can be less than authorized)
        ctx: Saga context
        
    Returns:
        Capture transaction details
    """
    
    await asyncio.sleep(0.05)
    
    return {
        "transaction_id": f"txn_{random.randint(100000, 999999)}",
        "authorization_id": authorization_id,
        "amount": amount,
        "status": "captured",
        "captured_at": "2024-12-15T10:00:00Z",
    }


async def process_wallet_payment(
    wallet_id: str,
    amount: float,
    currency: str = "USD",
    ctx: SagaContext = None
) -> Dict[str, Any]:
    """
    Process payment from digital wallet
    
    Args:
        wallet_id: Digital wallet identifier
        amount: Payment amount
        currency: Currency code
        ctx: Saga context
        
    Returns:
        Wallet transaction details
    """
    
    await asyncio.sleep(0.08)
    
    # Simulate insufficient funds 2% of the time
    if random.random() < 0.02:
        raise SagaStepError(f"Insufficient funds in wallet {wallet_id}")
    
    return {
        "transaction_id": f"wallet_{random.randint(100000, 999999)}",
        "wallet_id": wallet_id,
        "amount": amount,
        "currency": currency,
        "status": "completed",
        "processed_at": "2024-12-15T10:00:00Z",
    }


async def validate_payment_method(
    payment_method: Dict[str, Any] | str = None,
    ctx: SagaContext = None,
    user_id: str = None
) -> Dict[str, Any]:
    """
    Validate payment method before processing
    
    Args:
        payment_method: Payment method details (dict) or type (str)
        ctx: Saga context
        user_id: User ID for validation context
        
    Returns:
        Validation result
    """
    
    await asyncio.sleep(0.02)
    
    # Handle string input (payment method type)
    if isinstance(payment_method, str):
        payment_type = payment_method
        payment_method = {"type": payment_type}
    else:
        payment_type = payment_method.get("type")
    
    if payment_type == "credit_card":
        # For validation, we don't require the actual card token
        # Just validate that the type is supported
        pass
    elif payment_type == "wallet":
        # For validation, we don't require the actual wallet ID
        # Just validate that the type is supported  
        pass
    elif payment_type and payment_type not in ["credit_card", "wallet", "bank_transfer"]:
        raise SagaStepError(f"Unsupported payment method: {payment_type}")
    
    return {
        "valid": True,
        "payment_type": payment_type,
        "validated_at": "2024-12-15T10:00:00Z",
    }


# Convenience functions for tests and simple usage
async def process(user_id: str, amount: float, ctx: SagaContext) -> dict:
    """Process a payment - convenience wrapper around PaymentProvider.charge_card"""
    return await PaymentProvider.charge_card(
        card_token="test_token",
        amount=amount,
        currency="USD",
        metadata={"user_id": user_id}
    )


async def process_with_provider(provider: str, amount: float, payment_data: dict, ctx: SagaContext) -> dict:
    """Process payment with specific provider - for testing provider fallback logic"""
    if provider == "stripe":
        return await PaymentProvider.charge_card(
            card_token=payment_data.get("card_token", "test_token"),
            amount=amount,
            currency="USD",
            metadata={"provider": provider}
        )
    elif provider == "paypal":
        return await PaymentProvider.charge_wallet(
            wallet_id=payment_data.get("wallet_id", "test_wallet"),
            amount=amount,
            currency="USD"
        )
    else:
        raise ValueError(f"Unsupported payment provider: {provider}")