"""
API v1 router.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import agents, tools, workflows

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(agents.router, prefix="/agents", tags=["agents"])
api_router.include_router(workflows.router, prefix="/workflows", tags=["workflows"])
api_router.include_router(tools.router, prefix="/tools", tags=["tools"])
