"""
Structured logging for observability.

Provides context-aware logging with structured output format.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, TextIO

from .context import ContextManager, TraceContext


class LogLevel(str, Enum):
    """Log levels matching Python logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    
    def to_python_level(self) -> int:
        """Convert to Python logging level."""
        return getattr(logging, self.value)


@dataclass
class LogRecord:
    """A structured log record."""
    
    timestamp: datetime = field(default_factory=datetime.now)
    level: LogLevel = LogLevel.INFO
    message: str = ""
    logger_name: str = ""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        result = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "logger": self.logger_name,
        }
        
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.span_id:
            result["span_id"] = self.span_id
        if self.attributes:
            result["attributes"] = self.attributes
        if self.exception:
            result["exception"] = self.exception
        
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    def to_text(self) -> str:
        """Convert to human-readable text."""
        parts = [
            self.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            f"[{self.level.value}]",
            self.logger_name,
            "-",
            self.message,
        ]
        
        if self.trace_id:
            parts.append(f"(trace={self.trace_id[:8]})")
        
        if self.attributes:
            attrs = " ".join(f"{k}={v}" for k, v in self.attributes.items())
            parts.append(f"[{attrs}]")
        
        text = " ".join(parts)
        
        if self.exception:
            text += f"\n{self.exception}"
        
        return text


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""
    
    def __init__(self, json_output: bool = True):
        super().__init__()
        self.json_output = json_output
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record."""
        # Get trace context if available
        context = ContextManager.get_current()
        
        log_record = LogRecord(
            timestamp=datetime.fromtimestamp(record.created),
            level=LogLevel(record.levelname),
            message=record.getMessage(),
            logger_name=record.name,
            trace_id=context.trace_id if context else None,
            span_id=context.span_id if context else None,
            attributes=getattr(record, "attributes", {}),
            exception=self.formatException(record.exc_info) if record.exc_info else None,
        )
        
        if self.json_output:
            return log_record.to_json()
        return log_record.to_text()


class StructuredLogger:
    """
    Context-aware structured logger.
    
    Automatically includes trace context in log output and supports
    both JSON and text output formats.
    
    Usage:
        logger = StructuredLogger("my_module")
        
        logger.info("Processing started", agent="IssueAnalyzer", items=5)
        logger.error("Failed to process", exception=e, agent="IssueAnalyzer")
    """
    
    def __init__(
        self,
        name: str = "agentic_chain",
        level: LogLevel = LogLevel.INFO,
        json_output: bool = True,
        output: Optional[TextIO] = None,
    ):
        self.name = name
        self.level = level
        self.json_output = json_output
        self.output = output or sys.stdout
        
        # Set up Python logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.to_python_level())
        
        # Add handler if not already configured
        if not self._logger.handlers:
            handler = logging.StreamHandler(self.output)
            handler.setFormatter(StructuredFormatter(json_output))
            self._logger.addHandler(handler)
    
    def _log(
        self,
        level: LogLevel,
        message: str,
        exception: Optional[Exception] = None,
        **attributes,
    ) -> None:
        """Internal log method."""
        # Create extra data for the record
        extra = {"attributes": attributes}
        
        exc_info = None
        if exception:
            exc_info = (type(exception), exception, exception.__traceback__)
        
        self._logger.log(
            level.to_python_level(),
            message,
            exc_info=exc_info,
            extra=extra,
        )
    
    def debug(self, message: str, **attributes) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **attributes)
    
    def info(self, message: str, **attributes) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, **attributes)
    
    def warning(self, message: str, **attributes) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **attributes)
    
    def error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        **attributes,
    ) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, message, exception=exception, **attributes)
    
    def critical(
        self,
        message: str,
        exception: Optional[Exception] = None,
        **attributes,
    ) -> None:
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, exception=exception, **attributes)
    
    def with_context(self, **attributes) -> "ContextLogger":
        """Create a logger with additional context."""
        return ContextLogger(self, attributes)
    
    def set_level(self, level: LogLevel) -> None:
        """Set the log level."""
        self.level = level
        self._logger.setLevel(level.to_python_level())
    
    def get_child(self, suffix: str) -> "StructuredLogger":
        """Get a child logger with a suffix."""
        return StructuredLogger(
            name=f"{self.name}.{suffix}",
            level=self.level,
            json_output=self.json_output,
            output=self.output,
        )


class ContextLogger:
    """Logger wrapper that adds context to all messages."""
    
    def __init__(self, logger: StructuredLogger, context: Dict[str, Any]):
        self._logger = logger
        self._context = context
    
    def _merge_attrs(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Merge context with additional attributes."""
        merged = self._context.copy()
        merged.update(attributes)
        return merged
    
    def debug(self, message: str, **attributes) -> None:
        self._logger.debug(message, **self._merge_attrs(attributes))
    
    def info(self, message: str, **attributes) -> None:
        self._logger.info(message, **self._merge_attrs(attributes))
    
    def warning(self, message: str, **attributes) -> None:
        self._logger.warning(message, **self._merge_attrs(attributes))
    
    def error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        **attributes,
    ) -> None:
        self._logger.error(message, exception=exception, **self._merge_attrs(attributes))
    
    def critical(
        self,
        message: str,
        exception: Optional[Exception] = None,
        **attributes,
    ) -> None:
        self._logger.critical(message, exception=exception, **self._merge_attrs(attributes))


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    json_output: bool = True,
    output: Optional[TextIO] = None,
) -> StructuredLogger:
    """
    Configure and return the root structured logger.
    
    Args:
        level: Log level
        json_output: Whether to output JSON
        output: Output stream
        
    Returns:
        Configured StructuredLogger
    """
    return StructuredLogger(
        name="agentic_chain",
        level=level,
        json_output=json_output,
        output=output or sys.stdout,
    )
