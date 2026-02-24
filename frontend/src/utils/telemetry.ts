export type TelemetryLevel = 'info' | 'warning' | 'error';

interface ClientEventPayload {
  level: TelemetryLevel;
  event: string;
  details?: Record<string, unknown>;
}

export async function sendClientEvent(payload: ClientEventPayload): Promise<void> {
  try {
    await fetch('/api/v1/health/client-events', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      keepalive: true,
    });
  } catch {
    // Intentionally ignore telemetry transport failures.
  }
}
