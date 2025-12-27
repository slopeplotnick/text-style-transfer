"""
Thread-Safe Logger for Multi-threaded Processing

Provides thread-safe logging to prevent output mixing in parallel processing.
"""

import threading
import sys
from typing import Optional
from datetime import datetime
from queue import Queue
import time


class ThreadSafeLogger:
    """
    Thread-safe logger that queues messages and outputs them sequentially.
    """

    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize thread-safe logger.

        Args:
            log_file: Optional file path to save logs
        """
        self.lock = threading.Lock()
        self.log_file = log_file
        self.task_contexts = {}  # Store context for each thread

    def set_task_context(self, task_id: str, task_name: str):
        """
        Set context for current thread.

        Args:
            task_id: Unique identifier for the task
            task_name: Human-readable task name
        """
        thread_id = threading.current_thread().ident
        self.task_contexts[thread_id] = {
            'task_id': task_id,
            'task_name': task_name
        }

    def _get_prefix(self) -> str:
        """Get task-specific prefix for log messages."""
        thread_id = threading.current_thread().ident
        if thread_id in self.task_contexts:
            context = self.task_contexts[thread_id]
            return f"[{context['task_id']}] "
        return ""

    def log(self, message: str, level: str = "INFO"):
        """
        Log a message in a thread-safe manner.

        Args:
            message: The message to log
            level: Log level (INFO, WARNING, ERROR, etc.)
        """
        with self.lock:
            prefix = self._get_prefix()
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_msg = f"[{timestamp}] {prefix}{message}"

            # Print to stdout
            print(formatted_msg, flush=True)

            # Optionally write to file
            if self.log_file:
                try:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(formatted_msg + '\n')
                except Exception as e:
                    print(f"Failed to write to log file: {e}", file=sys.stderr)

    def info(self, message: str):
        """Log an info message."""
        self.log(message, "INFO")

    def warning(self, message: str):
        """Log a warning message."""
        self.log(f"[WARNING] {message}", "WARNING")

    def error(self, message: str):
        """Log an error message."""
        self.log(f"[ERROR] {message}", "ERROR")

    def success(self, message: str):
        """Log a success message."""
        self.log(f"✓ {message}", "SUCCESS")

    def section(self, title: str, width: int = 70):
        """Print a section header."""
        with self.lock:
            print("\n" + "=" * width)
            print(title)
            print("=" * width)

    def subsection(self, title: str):
        """Print a subsection header."""
        self.log(f"\n--- {title} ---")

    def progress(self, current: int, total: int, task_name: str):
        """
        Log progress information.

        Args:
            current: Current item number
            total: Total number of items
            task_name: Name of the task
        """
        percentage = (current / total) * 100 if total > 0 else 0
        self.log(f"[{current}/{total}] ({percentage:.1f}%) {task_name}")


# Global logger instance
_global_logger: Optional[ThreadSafeLogger] = None
_logger_lock = threading.Lock()


def get_logger(log_file: Optional[str] = None) -> ThreadSafeLogger:
    """
    Get global thread-safe logger instance.

    Args:
        log_file: Optional log file path

    Returns:
        ThreadSafeLogger instance
    """
    global _global_logger

    with _logger_lock:
        if _global_logger is None:
            _global_logger = ThreadSafeLogger(log_file)

    return _global_logger


def reset_logger():
    """Reset global logger instance."""
    global _global_logger

    with _logger_lock:
        _global_logger = None
