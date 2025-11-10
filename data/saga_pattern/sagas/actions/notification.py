"""
Notification actions for saga patterns

Reusable notification actions for email, SMS, and push notifications.
"""

import asyncio
import random
from typing import Any, Dict, List
from sage.core import SagaContext
from sage.exceptions import SagaStepError


class NotificationService:
    """Mock notification service for demonstration"""
    
    @staticmethod
    async def send_email(
        to: str,
        subject: str,
        body: str,
        template: str = None,
        template_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Send email notification"""
        
        await asyncio.sleep(0.05)  # Simulate API latency
        
        # Simulate 1% email delivery failure
        if random.random() < 0.01:
            raise Exception(f"Failed to send email to {to}")
        
        return {
            "message_id": f"email_{random.randint(100000, 999999)}",
            "to": to,
            "subject": subject,
            "status": "sent",
            "sent_at": "2024-12-15T10:00:00Z",
        }
    
    @staticmethod
    async def send_sms(
        phone: str,
        message: str
    ) -> Dict[str, Any]:
        """Send SMS notification"""
        
        await asyncio.sleep(0.03)
        
        # Simulate 2% SMS delivery failure
        if random.random() < 0.02:
            raise Exception(f"Failed to send SMS to {phone}")
        
        return {
            "message_id": f"sms_{random.randint(100000, 999999)}",
            "phone": phone,
            "status": "sent",
            "sent_at": "2024-12-15T10:00:00Z",
        }
    
    @staticmethod
    async def send_push(
        device_token: str,
        title: str,
        body: str,
        data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Send push notification"""
        
        await asyncio.sleep(0.02)
        
        return {
            "message_id": f"push_{random.randint(100000, 999999)}",
            "device_token": device_token,
            "title": title,
            "status": "sent",
            "sent_at": "2024-12-15T10:00:00Z",
        }


async def send_order_confirmation_email(
    user_email: str,
    order_details: Dict[str, Any],
    ctx: SagaContext = None
) -> Dict[str, Any]:
    """
    Send order confirmation email
    
    Args:
        user_email: Customer email address
        order_details: Order information
        ctx: Saga context
        
    Returns:
        Email delivery details
    """
    
    try:
        subject = f"Order Confirmation #{order_details.get('order_id', 'Unknown')}"
        
        body = f"""
        Thank you for your order!
        
        Order ID: {order_details.get('order_id')}
        Total: ${order_details.get('total', 0):.2f}
        Items: {len(order_details.get('items', []))}
        
        We'll send you updates as your order is processed.
        """
        
        result = await NotificationService.send_email(
            to=user_email,
            subject=subject,
            body=body,
            template="order_confirmation",
            template_data=order_details
        )
        
        return result
        
    except Exception as e:
        # Email failures are usually not critical - log but don't fail saga
        # In a real system, you might queue for retry
        return {
            "status": "failed",
            "error": str(e),
            "failed_at": "2024-12-15T10:00:00Z",
        }


async def send_payment_receipt_email(
    user_email: str,
    payment_details: Dict[str, Any],
    ctx: SagaContext = None
) -> Dict[str, Any]:
    """
    Send payment receipt email
    
    Args:
        user_email: Customer email address
        payment_details: Payment transaction details
        ctx: Saga context
        
    Returns:
        Email delivery details
    """
    
    try:
        subject = f"Payment Receipt - ${payment_details.get('amount', 0):.2f}"
        
        body = f"""
        Payment Receipt
        
        Transaction ID: {payment_details.get('transaction_id')}
        Amount: ${payment_details.get('amount', 0):.2f}
        Status: {payment_details.get('status', 'Unknown')}
        Date: {payment_details.get('processed_at')}
        """
        
        result = await NotificationService.send_email(
            to=user_email,
            subject=subject,
            body=body,
            template="payment_receipt",
            template_data=payment_details
        )
        
        return result
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "failed_at": "2024-12-15T10:00:00Z",
        }


async def send_shipping_notification(
    user_email: str,
    user_phone: str,
    shipping_details: Dict[str, Any],
    ctx: SagaContext = None
) -> Dict[str, Any]:
    """
    Send shipping notification via email and SMS
    
    Args:
        user_email: Customer email
        user_phone: Customer phone number
        shipping_details: Shipping information
        ctx: Saga context
        
    Returns:
        Notification delivery details
    """
    
    tracking_number = shipping_details.get('tracking_number')
    carrier = shipping_details.get('carrier', 'Our carrier')
    
    # Send email
    email_subject = f"Your order has shipped! Tracking: {tracking_number}"
    email_body = f"""
    Good news! Your order is on its way.
    
    Tracking Number: {tracking_number}
    Carrier: {carrier}
    Estimated Delivery: {shipping_details.get('estimated_delivery')}
    
    Track your package: {shipping_details.get('tracking_url')}
    """
    
    # Send SMS
    sms_message = f"{carrier}: Your order has shipped! Track with {tracking_number}"
    
    results = {}
    
    try:
        email_result = await NotificationService.send_email(
            to=user_email,
            subject=email_subject,
            body=email_body
        )
        results["email"] = email_result
    except Exception as e:
        results["email"] = {"status": "failed", "error": str(e)}
    
    try:
        sms_result = await NotificationService.send_sms(
            phone=user_phone,
            message=sms_message
        )
        results["sms"] = sms_result
    except Exception as e:
        results["sms"] = {"status": "failed", "error": str(e)}
    
    return results


async def send_order_cancellation_email(
    user_email: str,
    order_details: Dict[str, Any],
    reason: str = "Order cancelled",
    ctx: SagaContext = None
) -> Dict[str, Any]:
    """
    Send order cancellation email
    
    Args:
        user_email: Customer email address
        order_details: Order information
        reason: Cancellation reason
        ctx: Saga context
        
    Returns:
        Email delivery details
    """
    
    try:
        subject = f"Order Cancellation #{order_details.get('order_id')}"
        
        body = f"""
        Your order has been cancelled.
        
        Order ID: {order_details.get('order_id')}
        Reason: {reason}
        
        If you were charged, a refund will be processed within 3-5 business days.
        
        We apologize for any inconvenience.
        """
        
        result = await NotificationService.send_email(
            to=user_email,
            subject=subject,
            body=body,
            template="order_cancellation",
            template_data={**order_details, "reason": reason}
        )
        
        return result
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "failed_at": "2024-12-15T10:00:00Z",
        }


async def send_bulk_notifications(
    recipients: List[Dict[str, Any]],
    message_template: Dict[str, Any],
    ctx: SagaContext = None
) -> Dict[str, Any]:
    """
    Send bulk notifications to multiple recipients
    
    Args:
        recipients: List of recipient details
        message_template: Message template with placeholders
        ctx: Saga context
        
    Returns:
        Bulk send results
    """
    
    results = {
        "total": len(recipients),
        "successful": 0,
        "failed": 0,
        "details": []
    }
    
    for recipient in recipients:
        try:
            # Personalize message
            personalized_subject = message_template["subject"].format(**recipient)
            personalized_body = message_template["body"].format(**recipient)
            
            result = await NotificationService.send_email(
                to=recipient["email"],
                subject=personalized_subject,
                body=personalized_body
            )
            
            results["successful"] += 1
            results["details"].append({
                "recipient": recipient["email"],
                "status": "sent",
                "message_id": result["message_id"]
            })
            
        except Exception as e:
            results["failed"] += 1
            results["details"].append({
                "recipient": recipient["email"],
                "status": "failed",
                "error": str(e)
            })
    
    return results


# Convenience functions for tests and simple usage
async def send_email(to: str, subject: str, body: str, ctx: SagaContext) -> dict:
    """Send email - convenience wrapper around NotificationService.send_email"""
    result = await NotificationService.send_email(to, subject, body)
    result["sent"] = result.get("status") == "sent"
    return result


async def send_sms(to: str, message: str, ctx: SagaContext) -> dict:
    """Send SMS - convenience wrapper around NotificationService.send_sms"""
    result = await NotificationService.send_sms(to, message)
    result["sent"] = result.get("status") == "sent"
    return result