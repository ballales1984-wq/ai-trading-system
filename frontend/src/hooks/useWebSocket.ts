/**
 * useWebSocket — Generic WebSocket hook with auto-reconnect
 * Connects to the backend WS endpoint and provides:
 * - Auto-reconnect with exponential backoff
 * - Message parsing (JSON)
 * - Connection status
 */
import { useEffect, useRef, useCallback, useState } from 'react';

export type WsStatus = 'connecting' | 'open' | 'closed' | 'error';

interface UseWebSocketOptions {
  url: string;
  onMessage: (data: unknown) => void;
  enabled?: boolean;
  reconnectMaxAttempts?: number;
  reconnectBaseDelay?: number;
}

export function useWebSocket({
  url,
  onMessage,
  enabled = true,
  reconnectMaxAttempts = 5,
  reconnectBaseDelay = 1000,
}: UseWebSocketOptions) {
  const [status, setStatus] = useState<WsStatus>('closed');
  const wsRef = useRef<WebSocket | null>(null);
  const attemptsRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onMessageRef = useRef(onMessage);

  // Keep onMessage ref fresh without re-running the effect
  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus('connecting');
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus('open');
      attemptsRef.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data as string);
        onMessageRef.current(parsed);
      } catch {
        // Non-JSON frame — ignore
      }
    };

    ws.onerror = () => {
      setStatus('error');
    };

    ws.onclose = () => {
      setStatus('closed');
      wsRef.current = null;

      if (attemptsRef.current < reconnectMaxAttempts) {
        const delay = reconnectBaseDelay * 2 ** attemptsRef.current;
        attemptsRef.current += 1;
        reconnectTimerRef.current = setTimeout(() => {
          connect();
        }, delay);
      }
    };
  }, [url, reconnectMaxAttempts, reconnectBaseDelay]);

  useEffect(() => {
    if (!enabled) return;
    connect();

    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      // Prevent reconnect on intentional unmount
      attemptsRef.current = reconnectMaxAttempts;
      wsRef.current?.close();
    };
  }, [connect, enabled, reconnectMaxAttempts]);

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  return { status, send };
}
