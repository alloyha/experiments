"""
Payment compensation actions for saga patterns

Compensation actions to reverse payment operations when sagas fail.
"""

import asyncio
import logging
from typing import Any, Dict
from sage.core import SagaContext
from sage.exceptions import SagaCompensationError

logger = logging.getLogger(__name__)


async def refund_payment(
    payment_result: Dict[str, Any],
    ctx: SagaContext = None
) -> None:
    """
    Refund a completed payment transaction
    
    Args:
        payment_result: Result from process_payment action
        ctx: Saga context
    """
    
    if not payment_result or not payment_result.get("transaction_id"):
        # No payment to refund
        return
    
    try:
        transaction_id = payment_result["transaction_id"]
        amount = payment_result["amount"]
        
        # Import here to avoid circular imports
        from examples.actions.payment import PaymentProvider
        
        refund_result = await PaymentProvider.refund(
            transaction_id=transaction_id,
            amount=amount
        )
        
        # Log successful refund
        logger.info(
            f"Refunded payment {transaction_id}: ${amount} "
            f"(Refund ID: {refund_result['refund_id']})"
        )
    
    except Exception as e:
        raise SagaCompensationError(
            f"Failed to refund payment {payment_result.get('transaction_id')}: {str(e)}"
        )


async def void_authorization(
    auth_result: Dict[str, Any],
    ctx: SagaContext = None  
) -> None:
    """
    Void a payment authorization
    
    Args:
        auth_result: Result from authorize_payment action
        ctx: Saga context
    """
    
    if not auth_result or not auth_result.get("authorization_id"):
        return
    
    try:
        authorization_id = auth_result["authorization_id"]
        
        # Simulate voiding authorization
        await asyncio.sleep(0.05)
        
        logger.info(f"Voided payment authorization {authorization_id}")
    
    except Exception as e:
        raise SagaCompensationError(
            f"Failed to void authorization {auth_result.get('authorization_id')}: {str(e)}"
        )


async def reverse_wallet_payment(
    wallet_result: Dict[str, Any],
    ctx: SagaContext = None
) -> None:
    """
    Reverse a wallet payment transaction
    
    Args:
        wallet_result: Result from process_wallet_payment action
        ctx: Saga context
    """
    
    if not wallet_result or not wallet_result.get("transaction_id"):
        return
    
    try:
        transaction_id = wallet_result["transaction_id"]
        wallet_id = wallet_result["wallet_id"]
        amount = wallet_result["amount"]
        
        # Simulate reversing wallet transaction
        await asyncio.sleep(0.08)
        
        logger.info(
                f"Reversed wallet payment {transaction_id}: "
                f"${amount} returned to wallet {wallet_id}"
            )
    
    except Exception as e:
        raise SagaCompensationError(
            f"Failed to reverse wallet payment {wallet_result.get('transaction_id')}: {str(e)}"
        )


async def release_payment_hold(
    hold_result: Dict[str, Any],
    ctx: SagaContext = None
) -> None:
    """
    Release a payment hold/reserve
    
    Args:
        hold_result: Result from hold_payment action
        ctx: Saga context
    """
    
    if not hold_result or not hold_result.get("hold_id"):
        return
    
    try:
        hold_id = hold_result["hold_id"]
        amount = hold_result.get("amount", 0)
        
        # Simulate releasing payment hold
        await asyncio.sleep(0.03)
        
        logger.info(f"Released payment hold {hold_id}: ${amount}")
    
    except Exception as e:
        raise SagaCompensationError(
            f"Failed to release payment hold {hold_result.get('hold_id')}: {str(e)}"
        )


async def cancel_recurring_payment(
    recurring_result: Dict[str, Any],
    ctx: SagaContext = None
) -> None:
    """
    Cancel a recurring payment subscription
    
    Args:
        recurring_result: Result from setup_recurring_payment action
        ctx: Saga context
    """
    
    if not recurring_result or not recurring_result.get("subscription_id"):
        return
    
    try:
        subscription_id = recurring_result["subscription_id"]
        
        # Simulate canceling subscription
        await asyncio.sleep(0.05)
        
        logger.info(f"Cancelled recurring payment subscription {subscription_id}")
    
    except Exception as e:
        raise SagaCompensationError(
            f"Failed to cancel subscription {recurring_result.get('subscription_id')}: {str(e)}"
        )


async def reverse_payment_fee(
    fee_result: Dict[str, Any],
    ctx: SagaContext = None
) -> None:
    """
    Reverse/refund a payment processing fee
    
    Args:
        fee_result: Result from charge_processing_fee action
        ctx: Saga context
    """
    
    if not fee_result or not fee_result.get("fee_transaction_id"):
        return
    
    try:
        fee_transaction_id = fee_result["fee_transaction_id"]
        fee_amount = fee_result.get("fee_amount", 0)
        
        # Simulate reversing processing fee
        await asyncio.sleep(0.02)
        
        logger.info(f"Reversed processing fee {fee_transaction_id}: ${fee_amount}")
    
    except Exception as e:
        raise SagaCompensationError(
            f"Failed to reverse processing fee {fee_result.get('fee_transaction_id')}: {str(e)}"
        )


async def restore_payment_credits(
    credit_result: Dict[str, Any],
    ctx: SagaContext = None
) -> None:
    """
    Restore payment credits that were consumed
    
    Args:
        credit_result: Result from consume_payment_credits action
        ctx: Saga context
    """
    
    if not credit_result or not credit_result.get("credits_consumed"):
        return
    
    try:
        user_id = credit_result["user_id"]
        credits_consumed = credit_result["credits_consumed"]
        
        # Simulate restoring user credits
        await asyncio.sleep(0.02)
        
        logger.info(f"Restored {credits_consumed} credits to user {user_id}")
    
    except Exception as e:
        raise SagaCompensationError(
            f"Failed to restore credits for user {credit_result.get('user_id')}: {str(e)}"
        )


# Convenience functions for tests and simple usage
async def refund(payment_result: dict, ctx: SagaContext) -> None:
    """Refund payment - convenience wrapper around refund_payment"""
    await refund_payment(payment_result, ctx)


async def cancel_authorization(auth_result: dict, ctx: SagaContext) -> None:
    """Cancel authorization - convenience wrapper around void_authorization"""
    await void_authorization(auth_result, ctx)