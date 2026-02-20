# src/agents/base_agent.py
"""
Base Agent Module
=================
Abstract base class for all trading system agents.
Provides common interface and utilities for agent implementations.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from src.core.event_bus import EventBus, Event, EventType


logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent lifecycle states."""
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AgentMetrics:
    """Agent performance metrics."""
    events_processed: int = 0
    events_emitted: int = 0
    errors: int = 0
    last_activity: Optional[datetime] = None
    uptime_seconds: float = 0.0
    avg_processing_time_ms: float = 0.0


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.
    
    Provides:
    - Event bus integration
    - State management
    - Metrics tracking
    - Lifecycle management
    - Error handling
    """
    
    def __init__(
        self,
        name: str,
        event_bus: EventBus,
        state_manager: Any,
        config: Dict[str, Any]
    ):
        """
        Initialize the agent.
        
        Args:
            name: Unique agent identifier
            event_bus: Event bus for pub/sub communication
            state_manager: State manager for shared state
            config: Agent configuration dictionary
        """
        self.name = name
        self.event_bus = event_bus
        self.state_manager = state_manager
        self.config = config
        
        # Agent state
        self._state = AgentState.INITIALIZED
        self._running = False
        self._paused = False
        self._start_time: Optional[datetime] = None
        self._task: Optional[asyncio.Task] = None
        
        # Metrics
        self._metrics = AgentMetrics()
        
        # Error handling
        self._error_count = 0
        self._max_errors = config.get("max_errors", 10)
        self._error_backoff = config.get("error_backoff_seconds", 5.0)
        
        # Subscriptions
        self._subscriptions: list = []
        
        logger.info(f"Agent '{name}' initialized")
    
    @property
    def state(self) -> AgentState:
        """Get current agent state."""
        return self._state
    
    @property
    def metrics(self) -> AgentMetrics:
        """Get agent metrics."""
        return self._metrics
    
    async def start(self):
        """
        Start the agent.
        
        This method:
        1. Sets state to STARTING
        2. Calls on_start() hook
        3. Starts the main run loop
        4. Sets state to RUNNING
        """
        if self._running:
            logger.warning(f"Agent '{self.name}' is already running")
            return
        
        self._state = AgentState.STARTING
        self._start_time = datetime.now()
        
        try:
            # Hook for subclass initialization
            await self.on_start()
            
            # Subscribe to events
            self._setup_subscriptions()
            
            # Start main loop
            self._running = True
            self._state = AgentState.RUNNING
            self._task = asyncio.create_task(self._run_loop())
            
            # Emit started event
            await self.emit_event(
                EventType.ENGINE_STARTED,
                {"agent": self.name, "timestamp": datetime.now().isoformat()}
            )
            
            logger.info(f"Agent '{self.name}' started")
            
        except Exception as e:
            self._state = AgentState.ERROR
            logger.error(f"Failed to start agent '{self.name}': {e}")
            raise
    
    async def stop(self):
        """
        Stop the agent gracefully.
        
        This method:
        1. Sets state to STOPPING
        2. Cancels the main loop
        3. Calls on_stop() hook
        4. Sets state to STOPPED
        """
        if not self._running:
            return
        
        self._state = AgentState.STOPPING
        self._running = False
        
        try:
            # Cancel main task
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            
            # Unsubscribe from events
            self._teardown_subscriptions()
            
            # Hook for subclass cleanup
            await self.on_stop()
            
            self._state = AgentState.STOPPED
            
            # Emit stopped event
            await self.emit_event(
                EventType.ENGINE_STOPPED,
                {"agent": self.name, "timestamp": datetime.now().isoformat()}
            )
            
            logger.info(f"Agent '{self.name}' stopped")
            
        except Exception as e:
            self._state = AgentState.ERROR
            logger.error(f"Error stopping agent '{self.name}': {e}")
            raise
    
    async def pause(self):
        """Pause the agent."""
        self._paused = True
        self._state = AgentState.PAUSED
        logger.info(f"Agent '{self.name}' paused")
    
    async def resume(self):
        """Resume the agent."""
        self._paused = False
        self._state = AgentState.RUNNING
        logger.info(f"Agent '{self.name}' resumed")
    
    async def _run_loop(self):
        """
        Main run loop with error handling.
        """
        while self._running:
            try:
                if not self._paused:
                    await self.run()
                else:
                    await asyncio.sleep(0.1)
                    
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                self._metrics.errors += 1
                self._error_count += 1
                
                logger.error(f"Error in agent '{self.name}': {e}")
                
                if self._error_count >= self._max_errors:
                    logger.critical(
                        f"Agent '{self.name}' exceeded max errors ({self._max_errors})"
                    )
                    self._state = AgentState.ERROR
                    break
                
                # Backoff on error
                await asyncio.sleep(self._error_backoff)
    
    def _setup_subscriptions(self):
        """Setup event subscriptions."""
        for event_type, handler in self._subscriptions:
            self.event_bus.subscribe(event_type, handler)
    
    def _teardown_subscriptions(self):
        """Teardown event subscriptions."""
        for event_type, handler in self._subscriptions:
            self.event_bus.unsubscribe(event_type, handler)
    
    def subscribe_to(self, event_type: EventType, handler: Callable):
        """
        Subscribe to an event type.
        
        Args:
            event_type: Event type to subscribe to
            handler: Async or sync callback function
        """
        self._subscriptions.append((event_type, handler))
    
    async def emit_event(
        self,
        event_type: EventType,
        data: Dict[str, Any]
    ):
        """
        Emit an event to the event bus.
        
        Args:
            event_type: Type of event
            data: Event data dictionary
        """
        event = Event(
            event_type=event_type,
            data=data,
            source=self.name
        )
        await self.event_bus.publish(event)
        self._metrics.events_emitted += 1
    
    def update_state(self, key: str, value: Any):
        """
        Update shared state.
        
        Args:
            key: State key
            value: State value
        """
        self.state_manager.set(f"{self.name}:{key}", value)
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get shared state value.
        
        Args:
            key: State key
            default: Default value if not found
            
        Returns:
            State value or default
        """
        return self.state_manager.get(f"{self.name}:{key}", default)
    
    def get_shared_state(self, agent: str, key: str, default: Any = None) -> Any:
        """
        Get state from another agent.
        
        Args:
            agent: Agent name
            key: State key
            default: Default value
            
        Returns:
            State value or default
        """
        return self.state_manager.get(f"{agent}:{key}", default)
    
    @abstractmethod
    async def run(self):
        """
        Main agent logic - must be implemented by subclass.
        
        This method is called repeatedly while the agent is running.
        Implementations should:
        - Be non-blocking (use async/await)
        - Handle their own timing/sleeping
        - Update metrics as needed
        """
        pass
    
    async def on_start(self):
        """
        Hook called when agent starts.
        Override for custom initialization.
        """
        pass
    
    async def on_stop(self):
        """
        Hook called when agent stops.
        Override for custom cleanup.
        """
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent metrics as dictionary.
        
        Returns:
            Metrics dictionary
        """
        if self._start_time:
            self._metrics.uptime_seconds = (
                datetime.now() - self._start_time
            ).total_seconds()
        
        return {
            "name": self.name,
            "state": self._state.value,
            "events_processed": self._metrics.events_processed,
            "events_emitted": self._metrics.events_emitted,
            "errors": self._metrics.errors,
            "uptime_seconds": self._metrics.uptime_seconds,
            "last_activity": (
                self._metrics.last_activity.isoformat() 
                if self._metrics.last_activity else None
            ),
        }
