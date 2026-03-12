export function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    FILLED: 'bg-success/20 text-success',
    COMPLETED: 'bg-success/20 text-success',
    PENDING: 'bg-warning/20 text-warning',
    PARTIALLY_FILLED: 'bg-warning/20 text-warning',
    CANCELLED: 'bg-danger/20 text-danger',
    REJECTED: 'bg-danger/20 text-danger',
  };

  const style = styles[status] || 'bg-text-muted/20 text-text-muted';
  
  // Map status to accessible label
  const statusLabels: Record<string, string> = {
    FILLED: 'Order filled',
    COMPLETED: 'Order completed',
    PENDING: 'Order pending',
    PARTIALLY_FILLED: 'Order partially filled',
    CANCELLED: 'Order cancelled',
    REJECTED: 'Order rejected',
  };

  const accessibleLabel = statusLabels[status] || status;

  return (
    <span 
      className={`px-2 py-1 rounded text-xs font-medium ${style}`}
      role="status"
      aria-live="polite"
      title={accessibleLabel}
    >
      {status}
    </span>
  );
}