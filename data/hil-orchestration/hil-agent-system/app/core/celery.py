"""
Celery configuration for background tasks.
"""

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

# Create Celery app
celery_app = Celery(
    "hil-agent-system",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.agents.tasks",
        "app.workflows.tasks",
        "app.tools.tasks",
    ],
)

# Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_reject_on_worker_lost=True,
    worker_hijack_root_logger=False,
    worker_log_color=False,
    # Task routing
    task_routes={
        "app.agents.tasks.*": {"queue": "agents"},
        "app.workflows.tasks.*": {"queue": "workflows"},
        "app.tools.tasks.*": {"queue": "tools"},
    },
    # Task execution
    task_time_limit=1800,  # 30 minutes
    task_soft_time_limit=1500,  # 25 minutes
    worker_prefetch_multiplier=1,
    # Result backend
    result_expires=3600,  # 1 hour
    result_backend_max_retries=10,
    result_backend_retry_delay=1,
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Auto-discover tasks
celery_app.autodiscover_tasks()
