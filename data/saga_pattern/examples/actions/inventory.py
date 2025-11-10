# ============================================
# FILE: sagas/actions/inventory.py
# ============================================

import asyncio
from sage import SagaContext


async def reserve(items: list, ctx: SagaContext) -> dict:
    """
    Reserve inventory for items
    
    This is a reusable action that can be used by any saga
    """
    # Call your inventory service
    await asyncio.sleep(0.1)  # Simulate API call
    
    reservations = []
    for item in items:
        reservations.append({
            "item_id": item["id"],
            "quantity": item["quantity"],
            "reservation_id": f"RES-{item['id']}"
        })
    
    return {
        "reservations": reservations,
        "reserved_at": "2024-12-15T10:00:00Z"
    }


async def check_availability(items: list, ctx: SagaContext) -> dict:
    """Check if items are available"""
    # Your logic
    return {"available": True}