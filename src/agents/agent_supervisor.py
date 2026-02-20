# src/agents/agent_supervisor.py
"""
Supervisor Agent
================
Orchestrates all agents and manages the trading workflow.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from src.agents.base_agent import BaseAgent, AgentState
from src.core.event_bus import EventBus, EventType


logger = logging.getLogger(__name__)


class SystemMode(Enum):
    """System operating modes."""
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"
    BACKTEST = "backtest"
    ANALYSIS_ONLY = "analysis_only"


@dataclass
class SystemStatus:
    """System status container."""
    mode: SystemMode
    agents_running: int
    agents_healthy: int
    last_signal: Optional[str]
    last_trade: Optional[str]
    portfolio_value: float
    daily_pnl: float
    timestamp: datetime


class SupervisorAgent(BaseAgent):
    """
    Supervisor agent that orchestrates all other agents.
    
    Responsibilities:
    - Start/stop agents
    - Monitor agent health
    - Route signals to execution
    - Manage system state
    - Handle errors and recovery
    """
    
    def __init__(
        self,
        name: str,
        event_bus: EventBus,
        state_manager: Any,
        config: Dict[str, Any]
    ):
        """
        Initialize Supervisor Agent.
        
        Args:
            name: Agent identifier
            event_bus: Event bus for communication
            state_manager: State manager instance
            config: Configuration dictionary with:
                - mode: System operating mode
                - agents: List of agent instances to manage
                - health_check_interval: Seconds between health checks
                - max_restart_attempts: Max agent restart attempts
        """
        super().__init__(name, event_bus, state_manager, config)
        
        # System mode
        self.mode = SystemMode(config.get("mode", "paper_trading"))
        
        # Managed agents
        self._agents: List[BaseAgent] = config.get("agents", [])
        self._agent_status: Dict[str, AgentState] = {}
        
        # Health monitoring
        self.health_check_interval = config.get("health_check_interval", 30)
        self._max_restart_attempts = config.get("max_restart_attempts", 3)
        self._restart_counts: Dict[str, int] = {}
        
        # Trading state
        self._last_signal: Optional[str] = None
        self._last_trade: Optional[str] = None
        self._portfolio_value: float = 0.0
        self._daily_pnl: float = 0.0
        
        # Signal queue
        self._signal_queue: asyncio.Queue = asyncio.Queue()
        
        logger.info(f"SupervisorAgent initialized in {self.mode.value} mode")
    
    async def on_start(self):
        """Initialize and start all managed agents."""
        # Subscribe to events
        self.subscribe_to(EventType.SIGNAL_GENERATED, self._on_signal)
        self.subscribe_to(EventType.ORDER_FILLED, self._on_trade)
        self.subscribe_to(EventType.ENGINE_ERROR, self._on_agent_error)
        
        # Start all managed agents
        for agent in self._agents:
            try:
                await agent.start()
                self._agent_status[agent.name] = AgentState.RUNNING
                logger.info(f"Started agent: {agent.name}")
            except Exception as e:
                logger.error(f"Failed to start agent {agent.name}: {e}")
                self._agent_status[agent.name] = AgentState.ERROR
    
    async def on_stop(self):
        """Stop all managed agents."""
        for agent in self._agents:
            try:
                await agent.stop()
                self._agent_status[agent.name] = AgentState.STOPPED
                logger.info(f"Stopped agent: {agent.name}")
            except Exception as e:
                logger.error(f"Error stopping agent {agent.name}: {e}")
    
    async def run(self):
        """Main supervisor loop."""
        while self._running:
            try:
                # Health check
                await self._health_check()
                
                # Process signals
                await self._process_signals()
                
                # Update system status
                await self._update_status()
                
                # Wait for next cycle
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in supervisor loop: {e}")
                self._metrics.errors += 1
                await asyncio.sleep(self._error_backoff)
    
    async def _health_check(self):
        """Check health of all managed agents."""
        for agent in self._agents:
            status = agent.state
            self._agent_status[agent.name] = status
            
            if status == AgentState.ERROR:
                await self._handle_unhealthy_agent(agent)
            elif status == AgentState.STOPPED and agent.name not in self._restart_counts:
                # Agent stopped unexpectedly
                await self._handle_unhealthy_agent(agent)
    
    async def _handle_unhealthy_agent(self, agent: BaseAgent):
        """Handle an unhealthy agent."""
        restart_count = self._restart_counts.get(agent.name, 0)
        
        if restart_count < self._max_restart_attempts:
            logger.warning(
                f"Attempting to restart agent {agent.name} "
                f"(attempt {restart_count + 1}/{self._max_restart_attempts})"
            )
            
            try:
                await agent.stop()
                await asyncio.sleep(1)
                await agent.start()
                
                self._restart_counts[agent.name] = restart_count + 1
                self._agent_status[agent.name] = agent.state
                
                logger.info(f"Agent {agent.name} restarted successfully")
                
            except Exception as e:
                logger.error(f"Failed to restart agent {agent.name}: {e}")
        else:
            logger.critical(
                f"Agent {agent.name} exceeded max restart attempts. "
                "Manual intervention required."
            )
            
            await self.emit_event(
                EventType.RISK_ALERT,
                {
                    "type": "agent_failure",
                    "agent": agent.name,
                    "restart_attempts": restart_count,
                    "severity": "critical",
                }
            )
    
    async def _process_signals(self):
        """Process pending signals from the queue."""
        while not self._signal_queue.empty():
            try:
                signal = await asyncio.wait_for(
                    self._signal_queue.get(),
                    timeout=1.0
                )
                await self._route_signal(signal)
            except asyncio.TimeoutError:
                break
    
    async def _route_signal(self, signal: Dict):
        """Route a signal to the appropriate handler."""
        symbol = signal.get("symbol")
        action = signal.get("action", "HOLD")
        confidence = signal.get("confidence", 0)
        
        logger.info(f"Routing signal: {symbol} {action} (confidence: {confidence})")
        
        # Store last signal
        self._last_signal = f"{symbol}:{action}"
        self.update_state("last_signal", self._last_signal)
        
        # Only execute in trading modes
        if self.mode == SystemMode.ANALYSIS_ONLY:
            logger.info("Analysis mode - signal not executed")
            return
        
        # Check risk before execution
        risk_metrics = self.get_shared_state(
            "RiskAgent",
            f"risk:{symbol}",
            {}
        )
        
        if risk_metrics.get("risk_level") == "critical":
            logger.warning(f"Signal rejected due to critical risk: {symbol}")
            await self.emit_event(
                EventType.SIGNAL_REJECTED,
                {"symbol": symbol, "reason": "critical_risk"}
            )
            return
        
        # Route to execution
        await self.emit_event(
            EventType.SIGNAL_EXECUTED,
            signal
        )
    
    async def _on_signal(self, event):
        """Handle signal event."""
        await self._signal_queue.put(event.data)
        self._metrics.events_processed += 1
    
    async def _on_trade(self, event):
        """Handle trade event."""
        self._last_trade = event.data.get("order_id")
        self.update_state("last_trade", self._last_trade)
        self._metrics.events_processed += 1
    
    async def _on_agent_error(self, event):
        """Handle agent error event."""
        agent_name = event.data.get("agent")
        error = event.data.get("error")
        
        logger.error(f"Agent error from {agent_name}: {error}")
        
        if agent_name in self._agent_status:
            self._agent_status[agent_name] = AgentState.ERROR
    
    async def _update_status(self):
        """Update system status in shared state."""
        status = SystemStatus(
            mode=self.mode,
            agents_running=sum(
                1 for s in self._agent_status.values()
                if s == AgentState.RUNNING
            ),
            agents_healthy=sum(
                1 for s in self._agent_status.values()
                if s in (AgentState.RUNNING, AgentState.PAUSED)
            ),
            last_signal=self._last_signal,
            last_trade=self._last_trade,
            portfolio_value=self._portfolio_value,
            daily_pnl=self._daily_pnl,
            timestamp=datetime.now(),
        )
        
        self.update_state("system_status", {
            "mode": status.mode.value,
            "agents_running": status.agents_running,
            "agents_healthy": status.agents_healthy,
            "last_signal": status.last_signal,
            "last_trade": status.last_trade,
            "portfolio_value": status.portfolio_value,
            "daily_pnl": status.daily_pnl,
            "timestamp": status.timestamp.isoformat(),
        })
    
    def add_agent(self, agent: BaseAgent):
        """Add an agent to be managed."""
        self._agents.append(agent)
        self._agent_status[agent.name] = AgentState.INITIALIZED
    
    def remove_agent(self, agent_name: str):
        """Remove an agent from management."""
        self._agents = [a for a in self._agents if a.name != agent_name]
        self._agent_status.pop(agent_name, None)
    
    def get_agent_status(self, agent_name: str) -> Optional[AgentState]:
        """Get status of a specific agent."""
        return self._agent_status.get(agent_name)
    
    def get_all_agent_status(self) -> Dict[str, AgentState]:
        """Get status of all agents."""
        return self._agent_status.copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get supervisor metrics."""
        metrics = super().get_metrics()
        metrics.update({
            "mode": self.mode.value,
            "agents_managed": len(self._agents),
            "agents_running": sum(
                1 for s in self._agent_status.values()
                if s == AgentState.RUNNING
            ),
            "pending_signals": self._signal_queue.qsize(),
        })
        return metrics
