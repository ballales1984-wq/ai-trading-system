"""
Telegram Notifier Module
Real-time notifications via Telegram Bot
"""

import logging
import time
from datetime import datetime
from typing import Optional, Dict, List

import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Telegram Bot notification handler.
    Sends real-time alerts for trading events.
    """
    
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool = True,
        rate_limit: int = 5,  # Max messages per minute
        parse_mode: str = "Markdown"
    ):
        """
        Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram Bot API token
            chat_id: Target chat ID
            enabled: Whether notifications are enabled
            rate_limit: Maximum messages per minute
            parse_mode: Message parse mode (Markdown, HTML)
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.rate_limit = rate_limit
        self.parse_mode = parse_mode
        
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
        self.message_timestamps: List[float] = []
        
        # Test connection
        if self.enabled and bot_token and chat_id:
            self._test_connection()
        
        logger.info(f"TelegramNotifier initialized (enabled={enabled})")
    
    def _test_connection(self):
        """Test the Telegram bot connection."""
        try:
            response = requests.get(
                f"{self.api_url}/getMe",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    bot_name = data.get('result', {}).get('first_name', 'Unknown')
                    logger.info(f"Telegram bot connected: {bot_name}")
                    return
        
        except Exception as e:
            logger.warning(f"Telegram connection test failed: {e}")
    
    def _can_send(self) -> bool:
        """Check if we can send a message (rate limiting)."""
        if not self.enabled:
            return False
        
        now = time.time()
        
        # Remove old timestamps (older than 60 seconds)
        self.message_timestamps = [
            ts for ts in self.message_timestamps 
            if now - ts < 60
        ]
        
        # Check rate limit
        if len(self.message_timestamps) >= self.rate_limit:
            return False
        
        self.message_timestamps.append(now)
        return True
    
    def send(
        self,
        message: str,
        disable_notification: bool = False,
        reply_markup: Optional[Dict] = None
    ) -> bool:
        """
        Send a message via Telegram.
        
        Args:
            message: Message text
            disable_notification: Send silently
            reply_markup: Optional keyboard markup
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False
        
        if not self._can_send():
            logger.debug("Rate limit reached, skipping message")
            return False
        
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram bot not configured")
            return False
        
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": self.parse_mode,
            "disable_notification": disable_notification
        }
        
        if reply_markup:
            payload["reply_markup"] = reply_markup
        
        try:
            response = requests.post(
                f"{self.api_url}/sendMessage",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    logger.debug(f"Telegram message sent: {message[:50]}...")
                    return True
                else:
                    logger.error(f"Telegram API error: {data.get('description')}")
            else:
                logger.error(f"Telegram HTTP error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            logger.error("Telegram request timeout")
        except requests.exceptions.RequestException as e:
            logger.error(f"Telegram request error: {e}")
        
        return False
    
    # ==================== NOTIFICATION METHODS ====================
    
    def send_signal_alert(
        self,
        symbol: str,
        signal: int,
        price: float,
        confidence: float = 1.0
    ):
        """Send BUY/SELL signal alert."""
        if signal == 1:
            emoji = "📈"
            action = "BUY"
            color = "🟢"
        elif signal == -1:
            emoji = "📉"
            action = "SELL"
            color = "🔴"
        else:
            return
        
        message = (
            f"{emoji} *{action} SIGNAL* {emoji}\\n\\n"
            f"• *Symbol:* `{symbol}`\\n"
            f"• *Price:* ${price:,.2f}\\n"
            f"• *Confidence:* {confidence:.1%}"
        )
        
        self.send(message)
    
    def send_volatility_alert(
        self,
        symbol: str,
        volatility: float,
        threshold: float = 0.05
    ):
        """Send volatility alert."""
        if volatility > threshold:
            message = (
                f"⚠️ *HIGH VOLATILITY ALERT* ⚠️\\n\\n"
                f"• *Symbol:* `{symbol}`\\n"
                f"• *Volatility:* {volatility:.3f} ({volatility*100:.1f}%)\\n"
                f"• *Threshold:* {threshold:.3f}"
            )
            self.send(message)
    
    def send_regime_change(
        self,
        symbol: str,
        old_regime: str,
        new_regime: str
    ):
        """Send market regime change alert."""
        emoji_map = {
            'trending_up': '🚀',
            'trending_down': '📉',
            'ranging': '➡️',
            'volatile': '⚡',
            'neutral': '⚪'
        }
        
        emoji = emoji_map.get(new_regime, '⚪')
        
        message = (
            f"🔄 *REGIME CHANGE* 🔄\\n\\n"
            f"• *Symbol:* `{symbol}`\\n"
            f"• *Old Regime:* {old_regime}\\n"
            f"• *New Regime:* {emoji} {new_regime}"
        )
        
        self.send(message)
    
    def send_portfolio_update(
        self,
        total_value: float,
        total_pnl: float,
        positions: Dict[str, Dict]
    ):
        """Send portfolio status update."""
        pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
        
        message = (
            f"💰 *PORTFOLIO UPDATE* 💰\\n\\n"
            f"• *Total Value:* ${total_value:,.2f}\\n"
            f"• *Total P&L:* {pnl_emoji} ${total_pnl:+,.2f}\\n"
            f"• *Open Positions:* {len(positions)}\\n\\n"
        )
        
        # Add position details
        if positions:
            message += "*Positions:*\\n"
            for symbol, pos in positions.items():
                pnl = pos.get('unrealized_pnl', 0)
                pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                message += f"• `{symbol}`: {pnl_emoji} ${pnl:+,.2f}\\n"
        
        self.send(message)
    
    def send_trade_execution(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        pnl: Optional[float] = None
    ):
        """Send trade execution notification."""
        emoji = "🟢" if side.upper() == "BUY" else "🔴"
        
        message = (
            f"✅ *TRADE EXECUTED* ✅\\n\\n"
            f"• *Symbol:* `{symbol}`\\n"
            f"• *Side:* {emoji} {side.upper()}\\n"
            f"• *Quantity:* {quantity:.4f}\\n"
            f"• *Price:* ${price:,.2f}"
        )
        
        if pnl is not None:
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"
            message += f"\\n• *P&L:* {pnl_emoji} ${pnl:+,.2f}"
        
        self.send(message)
    
    def send_error_alert(
        self,
        error_type: str,
        message: str,
        details: Optional[str] = None
    ):
        """Send error alert."""
        msg = (
            f"❌ *ERROR ALERT* ❌\\n\\n"
            f"• *Type:* {error_type}\\n"
            f"• *Message:* {message}"
        )
        
        if details:
            msg += f"\\n• *Details:* `{details}`"
        
        self.send(msg)
    
    def send_status_update(
        self,
        status: str,
        uptime: Optional[str] = None
    ):
        """Send system status update."""
        message = (
            f"🔋 *SYSTEM STATUS* 🔋\\n\\n"
            f"• *Status:* {status}"
        )
        
        if uptime:
            message += f"\\n• *Uptime:* {uptime}"
        
        message += f"\\n• *Time:* {datetime.now().strftime('%H:%M:%S')}"
        
        self.send(message, disable_notification=True)
    
    def send_heartbeat(self, stats: Dict):
        """Send periodic heartbeat with stats."""
        message = (
            f"💓 *HEARTBEAT* 💓\\n\\n"
            f"• *Time:* {datetime.now().strftime('%H:%M:%S')}\\n"
            f"• *Total Value:* ${stats.get('total_value', 0):,.2f}\\n"
            f"• *P&L:* ${stats.get('total_pnl', 0):+,.2f}\\n"
            f"• *Positions:* {stats.get('num_positions', 0)}\\n"
            f"• *Trades Today:* {stats.get('trades_today', 0)}"
        )
        
        self.send(message, disable_notification=True)


class TelegramBotKeyboard:
    """Helper class for creating Telegram inline keyboards."""
    
    @staticmethod
    def create_url_button(text: str, url: str) -> Dict:
        """Create a URL button."""
        return {"text": text, "url": url}
    
    @staticmethod
    def create_callback_button(text: str, callback_data: str) -> Dict:
        """Create a callback button."""
        return {"text": text, "callback_data": callback_data}
    
    @staticmethod
    def create_keyboard(buttons: List[List[Dict]]) -> str:
        """Create inline keyboard markup."""
        return {"inline_keyboard": buttons}
    
    @staticmethod
    def create_trade_keyboard(symbol: str) -> str:
        """Create trade action keyboard."""
        buttons = [
            [
                TelegramBotKeyboard.create_callback_button("✅ CONFIRM", f"trade_{symbol}_confirm"),
                TelegramBotKeyboard.create_callback_button("❌ CANCEL", f"trade_{symbol}_cancel")
            ]
        ]
        return TelegramBotKeyboard.create_keyboard(buttons)
