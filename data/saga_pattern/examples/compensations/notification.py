"""
Notification compensation actions for saga patterns

Compensation actions for notification operations when sagas fail.
Note: Most notifications cannot be "undone" but we can send corrective messages.
"""

import asyncio
import logging
from typing import Any, Dict
from sage.core import SagaContext
from sage.exceptions import SagaCompensationError

logger = logging.getLogger(__name__)


async def send_order_cancellation_notification(
    notification_result: Dict[str, Any],
    ctx: SagaContext = None
) -> None:
    """
    Send cancellation notification after order confirmation was sent
    
    Args:
        notification_result: Result from send_order_confirmation_email
        ctx: Saga context
    """
    
    if not notification_result or notification_result.get("status") != "sent":
        # No notification to compensate for
        return
    
    try:
        # Import here to avoid circular imports
        from examples.actions.notification import send_order_cancellation_email
        
        # Extract order details from context
        order_details = ctx.get("order_details", {}) if ctx else {}
        user_email = ctx.get("user_email") if ctx else None
        
        if user_email and order_details:
            await send_order_cancellation_email(
                user_email=user_email,
                order_details=order_details,
                reason="Order processing failed",
                ctx=ctx
            )
            
            if ctx:
                logger.info(f"Sent order cancellation email to {user_email}")
    
    except Exception as e:
        # Notification failures during compensation are logged but not critical
        if ctx:
            logger.warning(f"Failed to send cancellation notification: {str(e)}")


async def send_payment_failure_notification(
    payment_notification_result: Dict[str, Any],
    ctx: SagaContext = None
) -> None:
    """
    Send payment failure notification after payment receipt was sent
    
    Args:
        payment_notification_result: Result from send_payment_receipt_email
        ctx: Saga context
    """
    
    if not payment_notification_result or payment_notification_result.get("status") != "sent":
        return
    
    try:
        from examples.actions.notification import NotificationService
        
        user_email = ctx.get("user_email") if ctx else None
        payment_details = ctx.get("payment_details", {}) if ctx else {}
        
        if user_email:
            subject = "Payment Issue - Action Required"
            body = f"""
            We encountered an issue processing your order after your payment was completed.
            
            Your payment of ${payment_details.get('amount', 0):.2f} will be refunded within 3-5 business days.
            
            Transaction ID: {payment_details.get('transaction_id')}
            
            We apologize for the inconvenience.
            """
            
            await NotificationService.send_email(
                to=user_email,
                subject=subject,
                body=body
            )
            
            if ctx:
                logger.info(f"Sent payment failure notification to {user_email}")
    
    except Exception as e:
        if ctx:
            logger.warning(f"Failed to send payment failure notification: {str(e)}")


async def send_shipping_cancellation_notification(
    shipping_notification_result: Dict[str, Any],
    ctx: SagaContext = None
) -> None:
    """
    Send shipping cancellation notification after shipping notification was sent
    
    Args:
        shipping_notification_result: Result from send_shipping_notification
        ctx: Saga context
    """
    
    if not shipping_notification_result:
        return
    
    # Check if any notifications were successfully sent
    email_sent = shipping_notification_result.get("email", {}).get("status") == "sent"
    sms_sent = shipping_notification_result.get("sms", {}).get("status") == "sent"
    
    if not (email_sent or sms_sent):
        return
    
    try:
        from examples.actions.notification import NotificationService
        
        user_email = ctx.get("user_email") if ctx else None
        user_phone = ctx.get("user_phone") if ctx else None
        order_details = ctx.get("order_details", {}) if ctx else {}
        
        # Send corrective email if original email was sent
        if email_sent and user_email:
            subject = f"Order Update - Shipment Delayed #{order_details.get('order_id')}"
            body = f"""
            We need to update you on your recent order.
            
            Your order #{order_details.get('order_id')} has encountered an issue and shipment has been delayed.
            
            We're working to resolve this and will contact you with an update soon.
            
            We apologize for any inconvenience.
            """
            
            await NotificationService.send_email(
                to=user_email,
                subject=subject,
                body=body
            )
        
        # Send corrective SMS if original SMS was sent
        if sms_sent and user_phone:
            message = f"Order #{order_details.get('order_id')} shipment delayed. We'll update you soon."
            
            await NotificationService.send_sms(
                phone=user_phone,
                message=message
            )
        
        if ctx:
            logger.info("Sent shipping cancellation notifications")
    
    except Exception as e:
        if ctx:
            logger.warning(f"Failed to send shipping cancellation notification: {str(e)}")


async def retract_bulk_notifications(
    bulk_result: Dict[str, Any],
    ctx: SagaContext = None
) -> None:
    """
    Send retraction message for bulk notifications
    
    Args:
        bulk_result: Result from send_bulk_notifications
        ctx: Saga context
    """
    
    if not bulk_result or bulk_result.get("successful", 0) == 0:
        return
    
    try:
        from examples.actions.notification import NotificationService
        
        # Get successfully sent notifications
        successful_recipients = [
            detail["recipient"] for detail in bulk_result.get("details", [])
            if detail.get("status") == "sent"
        ]
        
        if not successful_recipients:
            return
        
        # Send retraction email to each successful recipient
        retraction_subject = "Previous Message - Please Disregard"
        retraction_body = """
        Please disregard our previous message.
        
        This message was sent in error due to a system issue.
        
        We apologize for any confusion.
        """
        
        for recipient_email in successful_recipients:
            try:
                await NotificationService.send_email(
                    to=recipient_email,
                    subject=retraction_subject,
                    body=retraction_body
                )
            except Exception:
                # Continue with other recipients if one fails
                pass
        
        if ctx:
            logger.info(f"Sent retraction emails to {len(successful_recipients)} recipients")
    
    except Exception as e:
        if ctx:
            logger.warning(f"Failed to send bulk notification retractions: {str(e)}")


async def cancel_scheduled_notifications(
    schedule_result: Dict[str, Any],
    ctx: SagaContext = None
) -> None:
    """
    Cancel scheduled notifications that haven't been sent yet
    
    Args:
        schedule_result: Result from schedule_notification action
        ctx: Saga context
    """
    
    if not schedule_result or not schedule_result.get("schedule_id"):
        return
    
    try:
        schedule_id = schedule_result["schedule_id"]
        
        # Simulate canceling scheduled notification
        await asyncio.sleep(0.02)
        
        if ctx:
            logger.info(f"Cancelled scheduled notification {schedule_id}")
    
    except Exception as e:
        # Scheduled notification cancellation failure is not critical
        if ctx:
            logger.warning(f"Failed to cancel scheduled notification: {str(e)}")


async def suppress_notification_preferences(
    preference_result: Dict[str, Any],
    ctx: SagaContext = None
) -> None:
    """
    Restore notification preferences that were modified
    
    Args:
        preference_result: Result from modify_notification_preferences action
        ctx: Saga context
    """
    
    if not preference_result or not preference_result.get("user_id"):
        return
    
    try:
        user_id = preference_result["user_id"]
        original_preferences = preference_result.get("original_preferences", {})
        
        # Simulate restoring notification preferences
        await asyncio.sleep(0.02)
        
        if ctx:
            logger.info(f"Restored notification preferences for user {user_id}")
    
    except Exception as e:
        if ctx:
            logger.warning(f"Failed to restore notification preferences: {str(e)}")