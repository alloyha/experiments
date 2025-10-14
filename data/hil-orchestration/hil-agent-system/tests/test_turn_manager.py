"""
Tests for TurnManager service.
"""

import pytest
from uuid import uuid4

from app.models.conversation_turn import ConversationTurn
from app.services.turn_manager import TurnManager


@pytest.mark.asyncio
async def test_create_turn(db_session):
    """Test creating a new conversation turn."""
    turn_manager = TurnManager(db_session)
    conversation_id = uuid4()
    
    turn = await turn_manager.create_turn(
        conversation_id=conversation_id,
        session_id="test-session",
        turn_id="turn-001",
        user_input="Hello, I need help"
    )
    
    assert turn.id is not None
    assert turn.conversation_id == conversation_id
    assert turn.turn_number == 1
    assert turn.idempotency_key == "test-session:turn-001"
    assert turn.processing_status == "PROCESSING"
    assert turn.user_input == "Hello, I need help"
    assert turn.user_input_hash is not None


@pytest.mark.asyncio
async def test_idempotency_key_uniqueness(db_session):
    """Test that duplicate idempotency keys are rejected."""
    turn_manager = TurnManager(db_session)
    conversation_id = uuid4()
    
    # Create first turn
    turn1 = await turn_manager.create_turn(
        conversation_id=conversation_id,
        session_id="test-session",
        turn_id="turn-001",
        user_input="First message"
    )
    
    # Try to create duplicate
    with pytest.raises(ValueError, match="already exists"):
        await turn_manager.create_turn(
            conversation_id=conversation_id,
            session_id="test-session",
            turn_id="turn-001",  # Same turn_id
            user_input="Second message"
        )


@pytest.mark.asyncio
async def test_sequential_turn_numbering(db_session):
    """Test that turn numbers increment sequentially."""
    turn_manager = TurnManager(db_session)
    conversation_id = uuid4()
    
    # Create multiple turns
    turn1 = await turn_manager.create_turn(
        conversation_id=conversation_id,
        session_id="test-session",
        turn_id="turn-001",
        user_input="First message"
    )
    
    turn2 = await turn_manager.create_turn(
        conversation_id=conversation_id,
        session_id="test-session",
        turn_id="turn-002",
        user_input="Second message"
    )
    
    turn3 = await turn_manager.create_turn(
        conversation_id=conversation_id,
        session_id="test-session",
        turn_id="turn-003",
        user_input="Third message"
    )
    
    assert turn1.turn_number == 1
    assert turn2.turn_number == 2
    assert turn3.turn_number == 3


@pytest.mark.asyncio
async def test_complete_turn(db_session):
    """Test marking a turn as completed."""
    turn_manager = TurnManager(db_session)
    conversation_id = uuid4()
    
    # Create turn
    turn = await turn_manager.create_turn(
        conversation_id=conversation_id,
        session_id="test-session",
        turn_id="turn-001",
        user_input="Hello"
    )
    
    # Complete it
    execution_id = uuid4()
    completed_turn = await turn_manager.complete_turn(
        turn_id=turn.id,
        agent_response="Hi there! How can I help?",
        execution_id=execution_id
    )
    
    assert completed_turn.processing_status == "COMPLETED"
    assert completed_turn.agent_response == "Hi there! How can I help?"
    assert completed_turn.agent_response_hash is not None
    assert completed_turn.execution_id == execution_id
    assert completed_turn.completed_at is not None


@pytest.mark.asyncio
async def test_fail_turn(db_session):
    """Test marking a turn as failed."""
    turn_manager = TurnManager(db_session)
    conversation_id = uuid4()
    
    # Create turn
    turn = await turn_manager.create_turn(
        conversation_id=conversation_id,
        session_id="test-session",
        turn_id="turn-001",
        user_input="Hello"
    )
    
    # Fail it
    failed_turn = await turn_manager.fail_turn(
        turn_id=turn.id,
        error_message="LLM service unavailable"
    )
    
    assert failed_turn.processing_status == "FAILED"
    assert failed_turn.error_message == "LLM service unavailable"
    assert failed_turn.completed_at is not None


@pytest.mark.asyncio
async def test_get_by_idempotency_key(db_session):
    """Test retrieving turn by idempotency key."""
    turn_manager = TurnManager(db_session)
    conversation_id = uuid4()
    
    # Create turn
    turn = await turn_manager.create_turn(
        conversation_id=conversation_id,
        session_id="test-session",
        turn_id="turn-001",
        user_input="Hello"
    )
    
    # Retrieve by idempotency key
    retrieved = await turn_manager.get_by_idempotency_key("test-session:turn-001")
    
    assert retrieved is not None
    assert retrieved.id == turn.id
    assert retrieved.user_input == "Hello"


@pytest.mark.asyncio
async def test_check_duplicate_input(db_session):
    """Test duplicate input detection."""
    turn_manager = TurnManager(db_session)
    conversation_id = uuid4()
    
    # Create turn with specific input
    turn1 = await turn_manager.create_turn(
        conversation_id=conversation_id,
        session_id="test-session",
        turn_id="turn-001",
        user_input="I need help with my order"
    )
    await turn_manager.complete_turn(turn1.id, "Sure, what's your order number?")
    
    # Check for duplicate (exact same message)
    duplicate = await turn_manager.check_duplicate_input(
        conversation_id=conversation_id,
        user_input="I need help with my order"
    )
    
    assert duplicate is not None
    assert duplicate.id == turn1.id
    
    # Check for non-duplicate
    no_duplicate = await turn_manager.check_duplicate_input(
        conversation_id=conversation_id,
        user_input="Different message"
    )
    
    assert no_duplicate is None


@pytest.mark.asyncio
async def test_get_conversation_turns(db_session):
    """Test retrieving all turns for a conversation."""
    turn_manager = TurnManager(db_session)
    conversation_id = uuid4()
    
    # Create multiple turns
    for i in range(5):
        await turn_manager.create_turn(
            conversation_id=conversation_id,
            session_id="test-session",
            turn_id=f"turn-{i:03d}",
            user_input=f"Message {i}"
        )
    
    # Retrieve all turns
    turns = await turn_manager.get_conversation_turns(conversation_id)
    
    assert len(turns) == 5
    # Should be ordered by turn_number descending
    assert turns[0].turn_number == 5
    assert turns[4].turn_number == 1
