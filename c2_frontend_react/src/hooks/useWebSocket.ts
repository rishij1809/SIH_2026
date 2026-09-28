import { useEffect, useRef, useState, useCallback } from 'react';

export type WebSocketStatus = 'CONNECTING' | 'CONNECTED' | 'DISCONNECTED' | 'RECONNECTING';

export function useWebSocket<T>(
  endpoint: string,
  onMessage?: (data: T) => void
) {
  const [status, setStatus] = useState<WebSocketStatus>('CONNECTING');
  const [lastMessage, setLastMessage] = useState<T | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    const wsBase = import.meta.env.VITE_WS_BASE_URL || 
      (window.location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + window.location.host;

    const fullUrl = `${wsBase}${endpoint}`;
    setStatus('CONNECTING');

    try {
      const ws = new WebSocket(fullUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus('CONNECTED');
        console.log(`[WS CONNECTED] ${endpoint}`);
      };

      ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data) as T;
          setLastMessage(parsed);
          if (onMessage) {
            onMessage(parsed);
          }
        } catch (err) {
          console.error('[WS PARSE ERROR]', err);
        }
      };

      ws.onerror = (err) => {
        console.warn(`[WS ERROR] ${endpoint}`, err);
      };

      ws.onclose = () => {
        setStatus('RECONNECTING');
        wsRef.current = null;
        reconnectTimerRef.current = setTimeout(() => {
          connect();
        }, 3000);
      };
    } catch (err) {
      setStatus('DISCONNECTED');
    }
  }, [endpoint, onMessage]);

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  return { status, lastMessage };
}
