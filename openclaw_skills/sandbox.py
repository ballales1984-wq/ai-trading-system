"""
Sandbox Execution Module
========================
Provides a secure sandbox environment for executing untrusted Python code.

This module provides isolation for running user-provided or dynamically
generated code in a controlled environment with resource limits.

Note: For production use, consider using Docker containers or proper
sandboxing solutions like gVisor. This implementation provides basic
isolation through subprocess execution with timeouts.

Usage:
    from sandbox import run_sandboxed, SandboxError
    
    try:
        result = run_sandboxed("print('Hello from sandbox')", timeout_ms=1000)
    except SandboxError as e:
        print(f"Sandbox error: {e}")
    except TimeoutError:
        print("Execution timed out")
"""

import subprocess
import tempfile
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


class SandboxError(Exception):
    """
    Exception raised when sandbox execution fails.
    
    Attributes:
        message: Error message
        stderr: Standard error output from the subprocess
        return_code: Exit code of the subprocess
    """
    
    def __init__(self, message: str, stderr: str = "", return_code: int = -1):
        super().__init__(message)
        self.stderr = stderr
        self.return_code = return_code


@dataclass
class SandboxResult:
    """
    Result of sandboxed code execution.
    
    Attributes:
        stdout: Standard output from the executed code
        stderr: Standard error from the executed code
        return_code: Exit code of the subprocess
        execution_time_ms: Execution time in milliseconds
        success: Whether execution completed successfully
    """
    stdout: str
    stderr: str
    return_code: int
    execution_time_ms: float
    success: bool


def run_sandboxed(
    code: str,
    timeout_ms: int = 200,
    memory_limit_mb: int = 256,
    python_path: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None
) -> str:
    """
    Execute Python code in a sandboxed subprocess.
    
    This function creates a temporary Python script and executes it
    in a subprocess with specified resource limits.
    
    Args:
        code: Python code to execute
        timeout_ms: Maximum execution time in milliseconds
        memory_limit_mb: Memory limit in MB (informational - requires OS support)
        python_path: Path to Python interpreter (defaults to sys.executable)
        environment: Optional environment variables for the subprocess
        
    Returns:
        stdout: Standard output from the executed code
        
    Raises:
        SandboxError: If the code fails to execute or returns non-zero exit
        TimeoutError: If execution exceeds the timeout
        FileNotFoundError: If Python interpreter is not found
    
    Note:
        Memory limiting requires OS-level support (cgroups, ulimit, etc.)
        and is not enforced at the Python level.
    """
    # Use current Python interpreter if not specified
    if python_path is None:
        python_path = sys.executable
    
    # Verify Python interpreter exists
    if not Path(python_path).exists():
        raise FileNotFoundError(f"Python interpreter not found: {python_path}")
    
    # Create temporary directory for the script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "sandbox_script.py"
        
        # Write the code to a temporary file
        try:
            script_path.write_text(code, encoding="utf-8")
        except IOError as e:
            raise SandboxError(f"Failed to write sandbox script: {e}")
        
        # Prepare environment
        env = os.environ.copy()
        if environment:
            env.update(environment)
        
        # Set up execution
        try:
            proc = subprocess.run(
                [python_path, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_ms / 1000.0,  # Convert ms to seconds
                env=env,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(
                f"Sandbox execution exceeded timeout of {timeout_ms}ms"
            ) from e
        except OSError as e:
            raise SandboxError(f"Failed to execute sandbox: {e}")
        
        # Check for errors
        if proc.returncode != 0:
            raise SandboxError(
                f"Sandbox execution failed with exit code {proc.returncode}",
                stderr=proc.stderr.strip(),
                return_code=proc.returncode
            )
        
        return proc.stdout


def run_sandboxed_safe(
    code: str,
    timeout_ms: int = 200,
    memory_limit_mb: int = 256,
    allowed_modules: Optional[list[str]] = None,
    forbidden_modules: Optional[list[str]] = None
) -> SandboxResult:
    """
    Execute Python code in a sandboxed subprocess with detailed result tracking.
    
    This is a safer alternative that returns detailed execution information
    rather than raising exceptions.
    
    Args:
        code: Python code to execute
        timeout_ms: Maximum execution time in milliseconds
        memory_limit_mb: Memory limit in MB
        allowed_modules: List of modules that can be imported (not implemented)
        forbidden_modules: List of modules that cannot be imported (not implemented)
        
    Returns:
        SandboxResult: Detailed result of the execution
        
    Note:
        allowed_modules and forbidden_modules are not currently implemented.
        For proper module restrictions, use a proper sandboxing solution.
    """
    import time
    start_time = time.perf_counter()
    
    try:
        stdout = run_sandboxed(code, timeout_ms, memory_limit_mb)
        execution_time = (time.perf_counter() - start_time) * 1000
        
        return SandboxResult(
            stdout=stdout,
            stderr="",
            return_code=0,
            execution_time_ms=execution_time,
            success=True
        )
    except TimeoutError as e:
        execution_time = (time.perf_counter() - start_time) * 1000
        return SandboxResult(
            stdout="",
            stderr=str(e),
            return_code=-1,
            execution_time_ms=execution_time,
            success=False
        )
    except SandboxError as e:
        execution_time = (time.perf_counter() - start_time) * 1000
        return SandboxResult(
            stdout="",
            stderr=e.stderr or str(e),
            return_code=e.return_code,
            execution_time_ms=execution_time,
            success=False
        )
    except Exception as e:
        execution_time = (time.perf_counter() - start_time) * 1000
        return SandboxResult(
            stdout="",
            stderr=str(e),
            return_code=-1,
            execution_time_ms=execution_time,
            success=False
        )


def validate_code_safety(code: str) -> tuple[bool, list[str]]:
    """
    Perform basic static analysis to detect potentially dangerous code patterns.
    
    This is a heuristic check and should not be relied upon for security.
    Use proper sandboxing solutions for production systems.
    
    Args:
        code: Python code to validate
        
    Returns:
        Tuple of (is_safe, list_of_warnings)
    """
    warnings = []
    
    dangerous_patterns = [
        ("import os", "Module 'os' can access filesystem and processes"),
        ("import sys", "Module 'sys' can access Python runtime"),
        ("import subprocess", "Module 'subprocess' can execute commands"),
        ("import socket", "Module 'socket' can create network connections"),
        ("import requests", "Module 'requests' can make HTTP requests"),
        ("import urllib", "Module 'urllib' can make network requests"),
        ("open(", "File operations detected"),
        ("eval(", "Dynamic code evaluation detected"),
        ("exec(", "Dynamic code execution detected"),
        ("__import__", "Dynamic module import detected"),
        ("lambda", "Lambda functions may contain unsafe code"),
    ]
    
    for pattern, reason in dangerous_patterns:
        if pattern in code:
            warnings.append(f"Warning: {pattern} - {reason}")
    
    # Check for potentially infinite loops
    if "while True:" in code and "break" not in code:
        warnings.append("Warning: Potential infinite loop detected")
    
    return len(warnings) == 0, warnings


# Alias for backwards compatibility
SandboxRunner = run_sandboxed
