# ============================================
# FILE: sagas/compensations/inventory.py
# ============================================


import asyncio
from sage import SagaContext


async def release(reservation_result: dict, ctx: SagaContext) -> None:
    """
    Release inventory reservations
    
    Compensation for inventory.reserve()
    """
    await asyncio.sleep(0.1)  # Simulate API call
    
    for reservation in reservation_result["reservations"]:
        # Call your inventory service to release
        print(f"Released reservation: {reservation['reservation_id']}")