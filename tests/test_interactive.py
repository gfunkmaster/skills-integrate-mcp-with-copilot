"""
Tests for the Interactive Mode System.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from agentic_chain.interactive import (
    InteractionType,
    InteractionPoint,
    InteractionOption,
    InteractionResult,
    InteractionHistory,
    InteractionHandler,
    ConsoleInteractionHandler,
)
from agentic_chain.interactive.types import InteractionRecord
from agentic_chain.interactive.handler import AutoApproveHandler


class TestInteractionTypes:
    """Test interaction type definitions."""
    
    def test_interaction_option_creation(self):
        """Test creating an interaction option."""
        option = InteractionOption(
            id="opt1",
            label="Option 1",
            description="First option",
            risk_level="low",
            estimated_time="2 hours",
        )
        
        assert option.id == "opt1"
        assert option.label == "Option 1"
        assert option.risk_level == "low"
        assert option.estimated_time == "2 hours"
    
    def test_interaction_option_to_dict(self):
        """Test converting option to dictionary."""
        option = InteractionOption(
            id="opt1",
            label="Test Option",
            description="Description",
        )
        
        data = option.to_dict()
        
        assert data["id"] == "opt1"
        assert data["label"] == "Test Option"
        assert data["description"] == "Description"
    
    def test_interaction_option_from_dict(self):
        """Test creating option from dictionary."""
        data = {
            "id": "opt2",
            "label": "Option 2",
            "description": "Second option",
            "risk_level": "high",
        }
        
        option = InteractionOption.from_dict(data)
        
        assert option.id == "opt2"
        assert option.label == "Option 2"
        assert option.risk_level == "high"
    
    def test_interaction_point_creation(self):
        """Test creating an interaction point."""
        point = InteractionPoint(
            interaction_type=InteractionType.CONFIRMATION,
            title="Confirm Action",
            message="Do you want to proceed?",
        )
        
        assert point.interaction_type == InteractionType.CONFIRMATION
        assert point.title == "Confirm Action"
        assert point.allow_custom_input is True
    
    def test_interaction_point_with_options(self):
        """Test interaction point with options."""
        options = [
            InteractionOption(id="1", label="Option A"),
            InteractionOption(id="2", label="Option B"),
        ]
        
        point = InteractionPoint(
            interaction_type=InteractionType.ALTERNATIVE_SELECTION,
            title="Select an option",
            message="Choose one:",
            options=options,
        )
        
        assert len(point.options) == 2
        assert point.options[0].label == "Option A"
    
    def test_interaction_point_to_dict(self):
        """Test converting point to dictionary."""
        point = InteractionPoint(
            interaction_type=InteractionType.SOLUTION_REVIEW,
            title="Review Solution",
            message="Please review the proposed solution",
            context={"risk_level": "medium"},
        )
        
        data = point.to_dict()
        
        assert data["interaction_type"] == "solution_review"
        assert data["title"] == "Review Solution"
        assert data["context"]["risk_level"] == "medium"
    
    def test_interaction_point_from_dict(self):
        """Test creating point from dictionary."""
        data = {
            "interaction_type": "confirmation",
            "title": "Test",
            "message": "Test message",
            "options": [],
        }
        
        point = InteractionPoint.from_dict(data)
        
        assert point.interaction_type == InteractionType.CONFIRMATION
        assert point.title == "Test"
    
    def test_interaction_result_creation(self):
        """Test creating an interaction result."""
        result = InteractionResult(
            approved=True,
            selected_option="opt1",
            custom_input="Use Redis instead",
        )
        
        assert result.approved is True
        assert result.selected_option == "opt1"
        assert result.custom_input == "Use Redis instead"
        assert result.timestamp is not None
    
    def test_interaction_result_to_dict(self):
        """Test converting result to dictionary."""
        result = InteractionResult(
            approved=True,
            feedback="Good solution",
        )
        
        data = result.to_dict()
        
        assert data["approved"] is True
        assert data["feedback"] == "Good solution"
        assert data["timestamp"] is not None
    
    def test_interaction_result_from_dict(self):
        """Test creating result from dictionary."""
        data = {
            "approved": False,
            "selected_option": "opt2",
            "custom_input": None,
            "skipped": True,
        }
        
        result = InteractionResult.from_dict(data)
        
        assert result.approved is False
        assert result.selected_option == "opt2"
        assert result.skipped is True


class TestInteractionRecord:
    """Test interaction record functionality."""
    
    def test_record_creation(self):
        """Test creating an interaction record."""
        point = InteractionPoint(
            interaction_type=InteractionType.CONFIRMATION,
            title="Test",
            message="Test message",
        )
        result = InteractionResult(approved=True)
        
        record = InteractionRecord(
            point=point,
            result=result,
            agent_name="TestAgent",
        )
        
        assert record.id is not None
        assert record.point == point
        assert record.result == result
        assert record.agent_name == "TestAgent"
    
    def test_record_to_dict(self):
        """Test converting record to dictionary."""
        record = InteractionRecord(
            point=InteractionPoint(
                interaction_type=InteractionType.INPUT_REQUEST,
                title="Input",
                message="Enter input",
            ),
            result=InteractionResult(approved=True, custom_input="test"),
            agent_name="Agent1",
        )
        
        data = record.to_dict()
        
        assert data["agent_name"] == "Agent1"
        assert data["point"]["interaction_type"] == "input_request"
        assert data["result"]["custom_input"] == "test"


class TestInteractionHistory:
    """Test interaction history functionality."""
    
    def test_history_creation(self):
        """Test creating interaction history."""
        history = InteractionHistory()
        
        assert history.session_id is not None
        assert history.started_at is not None
        assert history.record_count == 0
    
    def test_add_record(self):
        """Test adding records to history."""
        history = InteractionHistory()
        
        record = InteractionRecord(
            point=InteractionPoint(
                interaction_type=InteractionType.CONFIRMATION,
                title="Test",
                message="Test",
            ),
            result=InteractionResult(approved=True),
        )
        
        history.add_record(record)
        
        assert history.record_count == 1
        assert history.approved_count == 1
        assert history.rejected_count == 0
    
    def test_history_counts(self):
        """Test history approval/rejection counts."""
        history = InteractionHistory()
        
        # Add approved record
        history.add_record(InteractionRecord(
            point=InteractionPoint(
                interaction_type=InteractionType.CONFIRMATION,
                title="Test1",
                message="Test1",
            ),
            result=InteractionResult(approved=True),
        ))
        
        # Add rejected record
        history.add_record(InteractionRecord(
            point=InteractionPoint(
                interaction_type=InteractionType.CONFIRMATION,
                title="Test2",
                message="Test2",
            ),
            result=InteractionResult(approved=False),
        ))
        
        assert history.record_count == 2
        assert history.approved_count == 1
        assert history.rejected_count == 1
    
    def test_history_complete(self):
        """Test completing a history session."""
        history = InteractionHistory()
        
        assert history.completed_at is None
        
        history.complete()
        
        assert history.completed_at is not None
    
    def test_history_to_dict(self):
        """Test converting history to dictionary."""
        history = InteractionHistory()
        history.metadata["test_key"] = "test_value"
        
        data = history.to_dict()
        
        assert "session_id" in data
        assert "started_at" in data
        assert data["record_count"] == 0
        assert data["metadata"]["test_key"] == "test_value"
    
    def test_history_to_json(self):
        """Test converting history to JSON string."""
        history = InteractionHistory()
        
        json_str = history.to_json()
        
        assert isinstance(json_str, str)
        assert "session_id" in json_str
    
    def test_history_from_json(self):
        """Test creating history from JSON string."""
        history = InteractionHistory()
        json_str = history.to_json()
        
        restored = InteractionHistory.from_json(json_str)
        
        assert restored.session_id == history.session_id


class TestConsoleInteractionHandler:
    """Test console interaction handler."""
    
    def test_handler_init(self):
        """Test handler initialization."""
        handler = ConsoleInteractionHandler(enabled=True)
        
        assert handler.enabled is True
        assert handler.history is not None
    
    def test_handler_disabled(self):
        """Test handler when disabled."""
        handler = ConsoleInteractionHandler(enabled=False)
        
        point = InteractionPoint(
            interaction_type=InteractionType.CONFIRMATION,
            title="Test",
            message="Test",
            default_option="yes",
        )
        
        result = handler.handle(point, agent_name="TestAgent")
        
        assert result.approved is True
        assert result.skipped is True
        assert result.selected_option == "yes"
    
    def test_handler_confirmation_yes(self):
        """Test confirmation with yes response."""
        mock_input = Mock(return_value="y")
        mock_output = Mock()
        
        handler = ConsoleInteractionHandler(
            enabled=True,
            input_func=mock_input,
            output_func=mock_output,
        )
        
        point = InteractionPoint(
            interaction_type=InteractionType.CONFIRMATION,
            title="Confirm",
            message="Proceed?",
            default_option="yes",
        )
        
        result = handler.handle(point, agent_name="TestAgent")
        
        assert result.approved is True
    
    def test_handler_confirmation_no(self):
        """Test confirmation with no response."""
        mock_input = Mock(return_value="n")
        mock_output = Mock()
        
        handler = ConsoleInteractionHandler(
            enabled=True,
            input_func=mock_input,
            output_func=mock_output,
        )
        
        point = InteractionPoint(
            interaction_type=InteractionType.CONFIRMATION,
            title="Confirm",
            message="Proceed?",
        )
        
        result = handler.handle(point, agent_name="TestAgent")
        
        assert result.approved is False
    
    def test_handler_selection(self):
        """Test selection interaction."""
        mock_input = Mock(return_value="1")
        mock_output = Mock()
        
        handler = ConsoleInteractionHandler(
            enabled=True,
            input_func=mock_input,
            output_func=mock_output,
        )
        
        options = [
            InteractionOption(id="opt1", label="Option 1"),
            InteractionOption(id="opt2", label="Option 2"),
        ]
        
        point = InteractionPoint(
            interaction_type=InteractionType.ALTERNATIVE_SELECTION,
            title="Select",
            message="Choose one:",
            options=options,
        )
        
        result = handler.handle(point, agent_name="TestAgent")
        
        assert result.approved is True
        assert result.selected_option == "opt1"
    
    def test_handler_custom_input(self):
        """Test custom input in selection."""
        mock_input = Mock(return_value="use redis instead")
        mock_output = Mock()
        
        handler = ConsoleInteractionHandler(
            enabled=True,
            input_func=mock_input,
            output_func=mock_output,
        )
        
        options = [
            InteractionOption(id="opt1", label="Use Memcached"),
        ]
        
        point = InteractionPoint(
            interaction_type=InteractionType.ALTERNATIVE_SELECTION,
            title="Select",
            message="Choose caching solution:",
            options=options,
            allow_custom_input=True,
        )
        
        result = handler.handle(point, agent_name="TestAgent")
        
        assert result.approved is True
        assert result.custom_input == "use redis instead"
    
    def test_handler_solution_review_approve(self):
        """Test solution review approval."""
        mock_input = Mock(return_value="y")
        mock_output = Mock()
        
        handler = ConsoleInteractionHandler(
            enabled=True,
            input_func=mock_input,
            output_func=mock_output,
        )
        
        point = InteractionPoint(
            interaction_type=InteractionType.SOLUTION_REVIEW,
            title="Review",
            message="Review the solution",
            context={
                "proposed_changes": [{"type": "modify", "description": "Update auth"}],
                "risks": [],
            },
        )
        
        result = handler.handle(point, agent_name="SolutionImplementer")
        
        assert result.approved is True
    
    def test_handler_solution_review_modify(self):
        """Test solution review with modifications."""
        mock_input = Mock(side_effect=["modify", "Use OAuth instead"])
        mock_output = Mock()
        
        handler = ConsoleInteractionHandler(
            enabled=True,
            input_func=mock_input,
            output_func=mock_output,
        )
        
        point = InteractionPoint(
            interaction_type=InteractionType.SOLUTION_REVIEW,
            title="Review",
            message="Review the solution",
            context={"proposed_changes": [], "risks": []},
        )
        
        result = handler.handle(point, agent_name="SolutionImplementer")
        
        assert result.approved is True
        assert result.feedback == "Use OAuth instead"
    
    def test_handler_records_history(self):
        """Test that handler records interaction history."""
        mock_input = Mock(return_value="y")
        mock_output = Mock()
        
        handler = ConsoleInteractionHandler(
            enabled=True,
            input_func=mock_input,
            output_func=mock_output,
        )
        
        point = InteractionPoint(
            interaction_type=InteractionType.CONFIRMATION,
            title="Test",
            message="Test",
        )
        
        handler.handle(point, agent_name="TestAgent")
        
        assert handler.history.record_count == 1
        assert handler.history.records[0].agent_name == "TestAgent"
    
    def test_handler_risk_assessment_high(self):
        """Test high risk assessment requires full 'yes'."""
        mock_input = Mock(return_value="y")  # Just 'y', not 'yes'
        mock_output = Mock()
        
        handler = ConsoleInteractionHandler(
            enabled=True,
            input_func=mock_input,
            output_func=mock_output,
        )
        
        point = InteractionPoint(
            interaction_type=InteractionType.RISK_ASSESSMENT,
            title="High Risk",
            message="This is risky",
            context={"risk_level": "high"},
        )
        
        result = handler.handle(point)
        
        # 'y' should not be enough for high risk
        assert result.approved is False
    
    def test_handler_risk_assessment_high_yes(self):
        """Test high risk assessment with full 'yes'."""
        mock_input = Mock(return_value="yes")
        mock_output = Mock()
        
        handler = ConsoleInteractionHandler(
            enabled=True,
            input_func=mock_input,
            output_func=mock_output,
        )
        
        point = InteractionPoint(
            interaction_type=InteractionType.RISK_ASSESSMENT,
            title="High Risk",
            message="This is risky",
            context={"risk_level": "high"},
        )
        
        result = handler.handle(point)
        
        assert result.approved is True
    
    def test_handler_keyboard_interrupt(self):
        """Test handling keyboard interrupt."""
        def raise_interrupt(*args):
            raise KeyboardInterrupt()
        
        mock_output = Mock()
        
        handler = ConsoleInteractionHandler(
            enabled=True,
            input_func=raise_interrupt,
            output_func=mock_output,
        )
        
        point = InteractionPoint(
            interaction_type=InteractionType.CONFIRMATION,
            title="Test",
            message="Test",
        )
        
        result = handler.handle(point)
        
        assert result.approved is False
        assert result.skipped is True
    
    def test_handler_request_confirmation(self):
        """Test request_confirmation convenience method."""
        mock_input = Mock(return_value="y")
        mock_output = Mock()
        
        handler = ConsoleInteractionHandler(
            enabled=True,
            input_func=mock_input,
            output_func=mock_output,
        )
        
        result = handler.request_confirmation(
            title="Confirm",
            message="Proceed?",
            agent_name="TestAgent",
        )
        
        assert result.approved is True
    
    def test_handler_request_selection(self):
        """Test request_selection convenience method."""
        mock_input = Mock(return_value="2")
        mock_output = Mock()
        
        handler = ConsoleInteractionHandler(
            enabled=True,
            input_func=mock_input,
            output_func=mock_output,
        )
        
        options = [
            InteractionOption(id="opt1", label="Option 1"),
            InteractionOption(id="opt2", label="Option 2"),
        ]
        
        result = handler.request_selection(
            title="Select",
            message="Choose:",
            options=options,
        )
        
        assert result.selected_option == "opt2"
    
    def test_handler_request_input(self):
        """Test request_input convenience method."""
        mock_input = Mock(return_value="user input text")
        mock_output = Mock()
        
        handler = ConsoleInteractionHandler(
            enabled=True,
            input_func=mock_input,
            output_func=mock_output,
        )
        
        result = handler.request_input(
            title="Input",
            message="Enter something:",
        )
        
        assert result.custom_input == "user input text"
    
    def test_reset_history(self):
        """Test resetting interaction history."""
        mock_input = Mock(return_value="y")
        mock_output = Mock()
        
        handler = ConsoleInteractionHandler(
            enabled=True,
            input_func=mock_input,
            output_func=mock_output,
        )
        
        # Add some interactions
        handler.request_confirmation("Test1", "Message1")
        handler.request_confirmation("Test2", "Message2")
        
        assert handler.history.record_count == 2
        
        # Reset
        handler.reset_history()
        
        assert handler.history.record_count == 0


class TestAutoApproveHandler:
    """Test auto-approve handler for automated mode."""
    
    def test_auto_approve_all(self):
        """Test that all interactions are auto-approved."""
        handler = AutoApproveHandler()
        
        point = InteractionPoint(
            interaction_type=InteractionType.RISK_ASSESSMENT,
            title="High Risk",
            message="Very risky operation",
            context={"risk_level": "high"},
        )
        
        result = handler.handle(point)
        
        assert result.approved is True
        assert result.skipped is True
    
    def test_auto_approve_selects_first_option(self):
        """Test that auto-approve selects first option."""
        handler = AutoApproveHandler()
        
        options = [
            InteractionOption(id="first", label="First"),
            InteractionOption(id="second", label="Second"),
        ]
        
        point = InteractionPoint(
            interaction_type=InteractionType.ALTERNATIVE_SELECTION,
            title="Select",
            message="Choose:",
            options=options,
        )
        
        result = handler.handle(point)
        
        assert result.selected_option == "first"
    
    def test_auto_approve_uses_default(self):
        """Test that auto-approve uses default option."""
        handler = AutoApproveHandler()
        
        options = [
            InteractionOption(id="first", label="First"),
            InteractionOption(id="second", label="Second"),
        ]
        
        point = InteractionPoint(
            interaction_type=InteractionType.ALTERNATIVE_SELECTION,
            title="Select",
            message="Choose:",
            options=options,
            default_option="second",
        )
        
        result = handler.handle(point)
        
        assert result.selected_option == "second"


class TestCallbackRegistration:
    """Test callback registration for interaction handlers."""
    
    def test_register_callback(self):
        """Test registering a callback."""
        handler = ConsoleInteractionHandler(enabled=False)
        
        callback_called = []
        def my_callback(point, result):
            callback_called.append((point, result))
        
        handler.register_callback(InteractionType.CONFIRMATION, my_callback)
        
        point = InteractionPoint(
            interaction_type=InteractionType.CONFIRMATION,
            title="Test",
            message="Test",
        )
        
        handler.handle(point)
        
        assert len(callback_called) == 1
        assert callback_called[0][0] == point
    
    def test_callback_error_handling(self):
        """Test that callback errors don't break handling."""
        handler = ConsoleInteractionHandler(enabled=False)
        
        def bad_callback(point, result):
            raise ValueError("Callback error")
        
        handler.register_callback(InteractionType.CONFIRMATION, bad_callback)
        
        point = InteractionPoint(
            interaction_type=InteractionType.CONFIRMATION,
            title="Test",
            message="Test",
        )
        
        # Should not raise
        result = handler.handle(point)
        
        assert result.approved is True
