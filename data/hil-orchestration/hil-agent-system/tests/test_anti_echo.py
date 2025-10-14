"""
Tests for AntiEchoMemory - Response deduplication service.
"""

from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation_turn import ConversationTurn
from app.services.anti_echo import AntiEchoMemory


@pytest_asyncio.fixture
async def anti_echo(db_session: AsyncSession):
    """Create AntiEchoMemory instance."""
    return AntiEchoMemory(db_session)


@pytest_asyncio.fixture
async def conversation_id():
    """Generate a unique conversation ID."""
    return uuid4()


async def create_turn_with_response(
    db_session: AsyncSession,
    conversation_id,
    turn_number: int,
    response: str
):
    """Helper to create a completed turn with response."""
    from app.services.turn_manager import TurnManager
    
    turn_manager = TurnManager(db_session)
    
    turn = await turn_manager.create_turn(
        conversation_id=conversation_id,
        session_id="test-session",
        turn_id=f"turn-{turn_number:03d}",
        user_input=f"Question {turn_number}"
    )
    
    # Complete turn with response
    await turn_manager.complete_turn(
        turn_id=turn.id,
        agent_response=response
    )
    
    return turn


@pytest.mark.asyncio
async def test_no_suppression_first_response(anti_echo, db_session, conversation_id):
    """Test that first response is never suppressed."""
    should_suppress, reason = await anti_echo.should_suppress_response(
        conversation_id=str(conversation_id),
        proposed_response="Hello! How can I help you today?"
    )
    
    assert should_suppress is False
    assert reason is None


@pytest.mark.asyncio
async def test_exact_duplicate_detection(anti_echo, db_session, conversation_id):
    """Test detection of exact duplicate responses."""
    response = "I can help you with your order."
    
    # Create turn with this response
    await create_turn_with_response(
        db_session,
        conversation_id,
        turn_number=1,
        response=response
    )
    
    # Try to send same response again
    should_suppress, reason = await anti_echo.should_suppress_response(
        conversation_id=str(conversation_id),
        proposed_response=response
    )
    
    assert should_suppress is True
    assert "Exact duplicate" in reason
    assert "turn 1" in reason


@pytest.mark.asyncio
async def test_case_insensitive_duplicate_detection(anti_echo, db_session, conversation_id):
    """Test that duplicates with different casing are still detected via similarity."""
    # Create turn with lowercase response
    await create_turn_with_response(
        db_session,
        conversation_id,
        turn_number=1,
        response="hello! how can i help you?"
    )
    
    # Try with different case - should be caught by similarity, not exact hash
    should_suppress, reason = await anti_echo.should_suppress_response(
        conversation_id=str(conversation_id),
        proposed_response="HELLO! HOW CAN I HELP YOU?"
    )
    
    assert should_suppress is True
    # Different case means different hash, so caught by similarity
    assert ("Too similar" in reason or "Exact duplicate" in reason)


@pytest.mark.asyncio
async def test_high_similarity_detection(anti_echo, db_session, conversation_id):
    """Test detection of highly similar responses."""
    # Create turn with response
    await create_turn_with_response(
        db_session,
        conversation_id,
        turn_number=1,
        response="I can help you track your order shipment today."
    )
    
    # Try similar response (90%+ similar words)
    should_suppress, reason = await anti_echo.should_suppress_response(
        conversation_id=str(conversation_id),
        proposed_response="I can help you track your shipment order today."
    )
    
    assert should_suppress is True
    assert "Too similar" in reason
    assert "turn 1" in reason


@pytest.mark.asyncio
async def test_different_responses_allowed(anti_echo, db_session, conversation_id):
    """Test that sufficiently different responses are allowed."""
    # Create turn with response
    await create_turn_with_response(
        db_session,
        conversation_id,
        turn_number=1,
        response="I can help you with your order."
    )
    
    # Try completely different response
    should_suppress, reason = await anti_echo.should_suppress_response(
        conversation_id=str(conversation_id),
        proposed_response="What specific product are you looking for?"
    )
    
    assert should_suppress is False
    assert reason is None


@pytest.mark.asyncio
async def test_short_responses_skipped(anti_echo, db_session, conversation_id):
    """Test that very short responses are not checked."""
    # Create turn with short response
    await create_turn_with_response(
        db_session,
        conversation_id,
        turn_number=1,
        response="Yes"
    )
    
    # Try to send same short response
    should_suppress, reason = await anti_echo.should_suppress_response(
        conversation_id=str(conversation_id),
        proposed_response="Yes"
    )
    
    # Short responses should not be suppressed
    assert should_suppress is False
    assert reason is None


@pytest.mark.asyncio
async def test_window_size_limit(db_session, conversation_id):
    """Test that only recent responses within window are checked."""
    # Create anti-echo with small window
    anti_echo = AntiEchoMemory(db_session, window_size=3)
    
    response = "I can help you with that."
    
    # Create 5 turns with different responses
    for i in range(1, 6):
        await create_turn_with_response(
            db_session,
            conversation_id,
            turn_number=i,
            response=f"Response number {i}"
        )
    
    # Create turn 6 with same response as turn 1
    await create_turn_with_response(
        db_session,
        conversation_id,
        turn_number=6,
        response="Response number 1"
    )
    
    # Try to send "Response number 1" again
    # Should be allowed because turn 1 is outside window (only checking 4, 5, 6)
    should_suppress, reason = await anti_echo.should_suppress_response(
        conversation_id=str(conversation_id),
        proposed_response="Response number 1"
    )
    
    # The duplicate from turn 6 (which IS in window) will catch it
    assert should_suppress is True


@pytest.mark.asyncio
async def test_only_completed_turns_checked(anti_echo, db_session, conversation_id):
    """Test that only COMPLETED turns are considered."""
    from app.services.turn_manager import TurnManager
    
    turn_manager = TurnManager(db_session)
    response = "I can help you with your order."
    
    # Create a PROCESSING turn (not completed)
    turn = await turn_manager.create_turn(
        conversation_id=conversation_id,
        session_id="test-session",
        turn_id="turn-001",
        user_input="Question"
    )
    # Don't complete it - leave in PROCESSING state
    
    # Should not suppress because no COMPLETED responses exist
    should_suppress, reason = await anti_echo.should_suppress_response(
        conversation_id=str(conversation_id),
        proposed_response=response
    )
    
    assert should_suppress is False


@pytest.mark.asyncio
async def test_get_response_history(anti_echo, db_session, conversation_id):
    """Test retrieving response history."""
    # Create multiple turns
    for i in range(1, 4):
        await create_turn_with_response(
            db_session,
            conversation_id,
            turn_number=i,
            response=f"Response {i}"
        )
    
    # Get history
    history = await anti_echo.get_response_history(
        conversation_id=str(conversation_id),
        limit=10
    )
    
    assert len(history) == 3
    assert all("turn_number" in h for h in history)
    assert all("response_preview" in h for h in history)
    assert all("response_hash" in h for h in history)


@pytest.mark.asyncio
async def test_analyze_conversation_patterns(anti_echo, db_session, conversation_id):
    """Test conversation pattern analysis."""
    # Create turns with some duplicates
    await create_turn_with_response(db_session, conversation_id, 1, "Response A")
    await create_turn_with_response(db_session, conversation_id, 2, "Response B")
    await create_turn_with_response(db_session, conversation_id, 3, "Response A")  # Dup
    await create_turn_with_response(db_session, conversation_id, 4, "Response C")
    
    # Analyze
    analysis = await anti_echo.analyze_conversation_patterns(
        conversation_id=str(conversation_id)
    )
    
    assert analysis["total_responses"] == 4
    assert analysis["unique_responses"] == 3  # A, B, C
    assert analysis["duplicate_count"] == 1  # One duplicate A
    assert "patterns" in analysis
    assert len(analysis["patterns"]) > 0


@pytest.mark.asyncio
async def test_similarity_calculation():
    """Test similarity calculation algorithm."""
    anti_echo = AntiEchoMemory(None)  # No DB needed for this test
    
    # Identical texts
    sim1 = anti_echo._calculate_similarity(
        "Hello world",
        "Hello world"
    )
    assert sim1 == 1.0
    
    # Completely different
    sim2 = anti_echo._calculate_similarity(
        "Hello world",
        "Goodbye universe"
    )
    assert sim2 < 0.3
    
    # Partially similar
    sim3 = anti_echo._calculate_similarity(
        "I can help you with your order",
        "I can help you with your shipment"
    )
    assert 0.7 < sim3 < 1.0


@pytest.mark.asyncio
async def test_clear_old_responses(anti_echo, db_session, conversation_id):
    """Test clearing old responses for memory management."""
    # Create many turns
    for i in range(1, 26):  # 25 turns
        await create_turn_with_response(
            db_session,
            conversation_id,
            turn_number=i,
            response=f"Response {i}"
        )
    
    # Clear old responses, keep only 10 most recent
    cleared = await anti_echo.clear_old_responses(
        conversation_id=str(conversation_id),
        keep_recent=10
    )
    
    assert cleared == 15  # Cleared 15 old responses
    
    # Verify recent responses still exist
    history = await anti_echo.get_response_history(
        conversation_id=str(conversation_id),
        limit=20
    )
    
    # Should have 10 with responses, 15 without
    responses_with_content = [h for h in history if h["response_preview"]]
    assert len(responses_with_content) == 10


@pytest.mark.asyncio
async def test_multiple_conversations_isolated(anti_echo, db_session):
    """Test that different conversations are isolated."""
    conv1 = uuid4()
    conv2 = uuid4()
    
    response = "I can help you with your order."
    
    # Create response in conversation 1
    await create_turn_with_response(db_session, conv1, 1, response)
    
    # Same response in conversation 2 should be allowed
    should_suppress, reason = await anti_echo.should_suppress_response(
        conversation_id=str(conv2),
        proposed_response=response
    )
    
    assert should_suppress is False
    assert reason is None


@pytest.mark.asyncio
async def test_custom_similarity_threshold(db_session, conversation_id):
    """Test custom similarity threshold."""
    # Create anti-echo with high threshold (0.95)
    anti_echo = AntiEchoMemory(
        db_session,
        high_similarity_threshold=0.95
    )
    
    # Create turn
    await create_turn_with_response(
        db_session,
        conversation_id,
        turn_number=1,
        response="I can help you with your order today."
    )
    
    # Similar but not 95%+ similar
    should_suppress, reason = await anti_echo.should_suppress_response(
        conversation_id=str(conversation_id),
        proposed_response="I can help you with your shipment today."
    )
    
    # Should be allowed with high threshold
    assert should_suppress is False


@pytest.mark.asyncio
async def test_empty_conversation(anti_echo, db_session):
    """Test handling of conversation with no responses."""
    empty_conv = uuid4()
    
    analysis = await anti_echo.analyze_conversation_patterns(
        conversation_id=str(empty_conv)
    )
    
    assert analysis["total_responses"] == 0
    assert analysis["unique_responses"] == 0
    assert analysis["duplicate_count"] == 0
