import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { emergencyApi, ordersApi } from '../services/api';
import { Plus, Play, X, Clock, CheckCircle, XCircle } from 'lucide-react';
import { TableSkeleton } from '../components/ui/Skeleton';
import { EmptyState, ErrorState } from '../components/ui/EmptyState';
import { formatCurrencyUSD, formatLocalDateTime } from '../utils/format';

export default function Orders() {
  const queryClient = useQueryClient();
  const [showForm, setShowForm] = useState(false);
  const [formError, setFormError] = useState('');
  const [symbolFilter, setSymbolFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [newOrder, setNewOrder] = useState({
    symbol: 'BTCUSDT',
    side: 'BUY',
    order_type: 'MARKET',
    quantity: 0.001,
    price: undefined as number | undefined,
    stop_price: undefined as number | undefined,
    broker: 'paper',
  });

  const { data: orders, isLoading: ordersLoading, error: ordersError } = useQuery({
    queryKey: ['orders', symbolFilter, statusFilter],
    queryFn: () => ordersApi.list(symbolFilter || undefined, statusFilter || undefined),
    refetchInterval: 10000,
  });

  const { data: emergencyStatus } = useQuery({
    queryKey: ['emergency-status'],
    queryFn: emergencyApi.getStatus,
  });

  const tradingHalted = Boolean(emergencyStatus?.trading_halted);

  const createOrder = useMutation({
    mutationFn: ordersApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['orders'] });
      setShowForm(false);
      setNewOrder({
        symbol: 'BTCUSDT',
        side: 'BUY',
        order_type: 'MARKET',
        quantity: 0.001,
        price: undefined,
        stop_price: undefined,
        broker: 'paper',
      });
      setFormError('');
    },
  });

  const cancelOrder = useMutation({
    mutationFn: ordersApi.cancel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['orders'] });
    },
  });

  const executeOrder = useMutation({
    mutationFn: ordersApi.execute,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['orders'] });
    },
  });

  const backendError =
    (createOrder.error as { response?: { data?: { detail?: string } } } | null)?.response?.data?.detail ||
    (cancelOrder.error as { response?: { data?: { detail?: string } } } | null)?.response?.data?.detail ||
    (executeOrder.error as { response?: { data?: { detail?: string } } } | null)?.response?.data?.detail ||
    '';

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (tradingHalted) return;
    if (newOrder.quantity <= 0 || Number.isNaN(newOrder.quantity)) {
      setFormError('Quantity must be greater than 0.');
      return;
    }
    if (newOrder.order_type === 'LIMIT' && !newOrder.price) {
      setFormError('Limit orders require a price.');
      return;
    }
    if (newOrder.order_type === 'STOP' && !newOrder.stop_price) {
      setFormError('Stop orders require a stop price.');
      return;
    }
    setFormError('');
    createOrder.mutate(newOrder);
  };

  const ordersList = Array.isArray(orders) ? orders : [];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'FILLED':
      case 'COMPLETED':
        return <CheckCircle className="w-4 h-4 text-success" />;
      case 'PENDING':
      case 'PARTIALLY_FILLED':
        return <Clock className="w-4 h-4 text-warning" />;
      case 'CANCELLED':
      case 'REJECTED':
        return <XCircle className="w-4 h-4 text-danger" />;
      default:
        return <Clock className="w-4 h-4 text-text-muted" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'FILLED':
      case 'COMPLETED':
        return 'text-success';
      case 'PENDING':
      case 'PARTIALLY_FILLED':
        return 'text-warning';
      case 'CANCELLED':
      case 'REJECTED':
        return 'text-danger';
      default:
        return 'text-text-muted';
    }
  };

  if (ordersError) {
    return (
      <div className="p-6">
        <ErrorState
          title="Failed to load orders"
          message="Unable to retrieve orders list from backend."
          retry={() => window.location.reload()}
        />
      </div>
    );
  }

  return (
    <div className="p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-text">Orders</h1>
          <p className="text-text-muted">Manage your trading orders</p>
        </div>
        <button
          onClick={() => setShowForm(!showForm)}
          disabled={tradingHalted}
          className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/80 transition-colors"
        >
          <Plus className="w-5 h-5" />
          New Order
        </button>
      </div>

      <div className="mb-4 grid grid-cols-1 gap-3 md:grid-cols-4">
        <input
          type="text"
          placeholder="Filter symbol (e.g. BTCUSDT)"
          value={symbolFilter}
          onChange={(e) => setSymbolFilter(e.target.value.toUpperCase())}
          className="rounded-lg border border-border bg-surface px-3 py-2 text-sm text-text"
        />
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="rounded-lg border border-border bg-surface px-3 py-2 text-sm text-text"
        >
          <option value="">All statuses</option>
          <option value="PENDING">PENDING</option>
          <option value="FILLED">FILLED</option>
          <option value="PARTIALLY_FILLED">PARTIALLY_FILLED</option>
          <option value="CANCELLED">CANCELLED</option>
          <option value="REJECTED">REJECTED</option>
        </select>
        <button
          onClick={() => queryClient.invalidateQueries({ queryKey: ['orders'] })}
          className="rounded-lg border border-border bg-surface px-3 py-2 text-sm text-text hover:bg-border/40"
        >
          Refresh orders
        </button>
      </div>

      {tradingHalted && (
        <div className="mb-4 rounded-lg border border-danger/50 bg-danger/10 px-4 py-3 text-danger">
          Emergency Stop attivo: creazione ed esecuzione ordini BUY/SELL bloccate.
        </div>
      )}

      {/* New Order Form */}
      {showForm && (
        <div className="bg-surface border border-border rounded-lg p-4 mb-6">
          <h2 className="text-lg font-semibold text-text mb-4">Create New Order</h2>
          {formError && (
            <div className="mb-4 rounded-lg border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-danger">
              {formError}
            </div>
          )}
          <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label htmlFor="order-symbol" className="block text-text-muted text-sm mb-1">Symbol</label>
              <select
                id="order-symbol"
                name="symbol"
                value={newOrder.symbol}
                onChange={(e) => setNewOrder({ ...newOrder, symbol: e.target.value })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text"
              >
                <option value="BTCUSDT">BTCUSDT</option>
                <option value="ETHUSDT">ETHUSDT</option>
                <option value="SOLUSDT">SOLUSDT</option>
                <option value="BNBUSDT">BNBUSDT</option>
                <option value="EURUSD">EURUSD</option>
              </select>
            </div>
            <div>
              <label htmlFor="order-side" className="block text-text-muted text-sm mb-1">Side</label>
              <select
                id="order-side"
                name="side"
                value={newOrder.side}
                onChange={(e) => setNewOrder({ ...newOrder, side: e.target.value })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text"
              >
                <option value="BUY">BUY</option>
                <option value="SELL">SELL</option>
              </select>
            </div>
            <div>
              <label htmlFor="order-type" className="block text-text-muted text-sm mb-1">Order Type</label>
              <select
                id="order-type"
                name="order_type"
                value={newOrder.order_type}
                onChange={(e) => setNewOrder({ ...newOrder, order_type: e.target.value })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text"
              >
                <option value="MARKET">Market</option>
                <option value="LIMIT">Limit</option>
                <option value="STOP">Stop</option>
              </select>
            </div>
            <div>
              <label htmlFor="order-quantity" className="block text-text-muted text-sm mb-1">Quantity</label>
              <input
                id="order-quantity"
                name="quantity"
                type="number"
                step="0.0001"
                autoComplete="off"
                value={newOrder.quantity}
                onChange={(e) => setNewOrder({ ...newOrder, quantity: parseFloat(e.target.value) })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text"
              />
            </div>
            {newOrder.order_type === 'LIMIT' && (
              <div>
                <label htmlFor="order-price" className="block text-text-muted text-sm mb-1">Limit Price</label>
                <input
                  id="order-price"
                  name="price"
                  type="number"
                  step="0.0001"
                  autoComplete="off"
                  value={newOrder.price ?? ''}
                  onChange={(e) =>
                    setNewOrder({
                      ...newOrder,
                      price: e.target.value ? parseFloat(e.target.value) : undefined,
                    })
                  }
                  className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text"
                />
              </div>
            )}
            {newOrder.order_type === 'STOP' && (
              <div>
                <label htmlFor="order-stop-price" className="block text-text-muted text-sm mb-1">Stop Price</label>
                <input
                  id="order-stop-price"
                  name="stop_price"
                  type="number"
                  step="0.0001"
                  autoComplete="off"
                  value={newOrder.stop_price ?? ''}
                  onChange={(e) =>
                    setNewOrder({
                      ...newOrder,
                      stop_price: e.target.value ? parseFloat(e.target.value) : undefined,
                    })
                  }
                  className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text"
                />
              </div>
            )}
            <div>
              <label htmlFor="order-broker" className="block text-text-muted text-sm mb-1">Broker</label>
              <select
                id="order-broker"
                name="broker"
                value={newOrder.broker}
                onChange={(e) => setNewOrder({ ...newOrder, broker: e.target.value })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text"
              >
                <option value="paper">Paper Trading</option>
                <option value="binance">Binance</option>
                <option value="bybit">Bybit</option>
                <option value="ib">Interactive Brokers</option>
              </select>
            </div>
            <div className="flex items-end gap-2">
              <button
                type="submit"
                disabled={createOrder.isPending || tradingHalted}
                className="flex-1 px-4 py-2 bg-success text-white rounded-lg hover:bg-success/80 transition-colors disabled:opacity-50"
              >
                {createOrder.isPending ? 'Creating...' : 'Create Order'}
              </button>
              <button
                type="button"
                onClick={() => setShowForm(false)}
                className="px-4 py-2 bg-surface border border-border text-text rounded-lg hover:bg-border/50 transition-colors"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {backendError && (
        <div className="mb-4 rounded-lg border border-danger/50 bg-danger/10 px-4 py-3 text-sm text-danger">
          {backendError}
        </div>
      )}

      {/* Orders Table */}
      <div className="bg-surface border border-border rounded-lg p-4">
        <h2 className="text-lg font-semibold text-text mb-4">All Orders</h2>
        <div className="overflow-x-auto">
          {ordersLoading ? (
            <TableSkeleton rows={6} />
          ) : (
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 px-4 text-text-muted font-medium">Order ID</th>
                <th className="text-left py-3 px-4 text-text-muted font-medium">Symbol</th>
                <th className="text-left py-3 px-4 text-text-muted font-medium">Side</th>
                <th className="text-left py-3 px-4 text-text-muted font-medium">Type</th>
                <th className="text-right py-3 px-4 text-text-muted font-medium">Quantity</th>
                <th className="text-right py-3 px-4 text-text-muted font-medium">Price</th>
                <th className="text-center py-3 px-4 text-text-muted font-medium">Status</th>
                <th className="text-left py-3 px-4 text-text-muted font-medium">Created</th>
                <th className="text-center py-3 px-4 text-text-muted font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {ordersList.map((order) => (
                <tr key={order.order_id} className="border-b border-border/50 hover:bg-border/20">
                  <td className="py-3 px-4 text-text-muted font-mono text-sm">{order.order_id.slice(0, 8)}...</td>
                  <td className="py-3 px-4 font-medium text-text">{order.symbol}</td>
                  <td className={`py-3 px-4 ${order.side === 'BUY' ? 'text-success' : 'text-danger'}`}>
                    {order.side}
                  </td>
                  <td className="py-3 px-4 text-text-muted">{order.order_type}</td>
                  <td className="py-3 px-4 text-right text-text">{order.quantity.toFixed(4)}</td>
                  <td className="py-3 px-4 text-right text-text">
                    {order.average_price ? formatCurrencyUSD(order.average_price) : '-'}
                  </td>
                  <td className="py-3 px-4 text-center">
                    <div className="flex items-center justify-center gap-2">
                      {getStatusIcon(order.status)}
                      <span className={getStatusColor(order.status)}>{order.status}</span>
                    </div>
                  </td>
                  <td className="py-3 px-4 text-text-muted text-sm">
                    {formatLocalDateTime(order.created_at)}
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center justify-center gap-2">
                      {order.status === 'PENDING' && (
                        <>
                          <button
                            onClick={() => executeOrder.mutate(order.order_id)}
                            disabled={tradingHalted}
                            className="p-1 text-success hover:bg-success/20 rounded"
                            title="Execute"
                          >
                            <Play className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => cancelOrder.mutate(order.order_id)}
                            className="p-1 text-danger hover:bg-danger/20 rounded"
                            title="Cancel"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        </>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          )}
          {!ordersLoading && ordersList.length === 0 && (
            <EmptyState
              icon={Clock}
              title="No orders found"
              description="Create your first order to start tracking execution."
            />
          )}
        </div>
      </div>
    </div>
  );
}

