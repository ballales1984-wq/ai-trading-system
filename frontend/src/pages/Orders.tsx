import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { ordersApi } from '../services/api';
import { Plus, Play, X, Clock, CheckCircle, XCircle } from 'lucide-react';

export default function Orders() {
  const queryClient = useQueryClient();
  const [showForm, setShowForm] = useState(false);
  const [newOrder, setNewOrder] = useState({
    symbol: 'BTCUSDT',
    side: 'BUY',
    order_type: 'MARKET',
    quantity: 0.001,
    broker: 'paper',
  });

  const { data: orders } = useQuery({
    queryKey: ['orders'],
    queryFn: () => ordersApi.list(),
    refetchInterval: 10000,
  });

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
        broker: 'paper',
      });
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

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    createOrder.mutate(newOrder);
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

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
          className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/80 transition-colors"
        >
          <Plus className="w-5 h-5" />
          New Order
        </button>
      </div>

      {/* New Order Form */}
      {showForm && (
        <div className="bg-surface border border-border rounded-lg p-4 mb-6">
          <h2 className="text-lg font-semibold text-text mb-4">Create New Order</h2>
          <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-text-muted text-sm mb-1">Symbol</label>
              <select
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
              <label className="block text-text-muted text-sm mb-1">Side</label>
              <select
                value={newOrder.side}
                onChange={(e) => setNewOrder({ ...newOrder, side: e.target.value })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text"
              >
                <option value="BUY">BUY</option>
                <option value="SELL">SELL</option>
              </select>
            </div>
            <div>
              <label className="block text-text-muted text-sm mb-1">Order Type</label>
              <select
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
              <label className="block text-text-muted text-sm mb-1">Quantity</label>
              <input
                type="number"
                step="0.0001"
                value={newOrder.quantity}
                onChange={(e) => setNewOrder({ ...newOrder, quantity: parseFloat(e.target.value) })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text"
              />
            </div>
            <div>
              <label className="block text-text-muted text-sm mb-1">Broker</label>
              <select
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
                disabled={createOrder.isPending}
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

      {/* Orders Table */}
      <div className="bg-surface border border-border rounded-lg p-4">
        <h2 className="text-lg font-semibold text-text mb-4">All Orders</h2>
        <div className="overflow-x-auto">
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
              {orders?.map((order) => (
                <tr key={order.order_id} className="border-b border-border/50 hover:bg-border/20">
                  <td className="py-3 px-4 text-text-muted font-mono text-sm">{order.order_id.slice(0, 8)}...</td>
                  <td className="py-3 px-4 font-medium text-text">{order.symbol}</td>
                  <td className={`py-3 px-4 ${order.side === 'BUY' ? 'text-success' : 'text-danger'}`}>
                    {order.side}
                  </td>
                  <td className="py-3 px-4 text-text-muted">{order.order_type}</td>
                  <td className="py-3 px-4 text-right text-text">{order.quantity.toFixed(4)}</td>
                  <td className="py-3 px-4 text-right text-text">
                    {order.average_price ? formatCurrency(order.average_price) : '-'}
                  </td>
                  <td className="py-3 px-4 text-center">
                    <div className="flex items-center justify-center gap-2">
                      {getStatusIcon(order.status)}
                      <span className={getStatusColor(order.status)}>{order.status}</span>
                    </div>
                  </td>
                  <td className="py-3 px-4 text-text-muted text-sm">
                    {new Date(order.created_at).toLocaleString()}
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center justify-center gap-2">
                      {order.status === 'PENDING' && (
                        <>
                          <button
                            onClick={() => executeOrder.mutate(order.order_id)}
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
          {orders?.length === 0 && (
            <div className="text-center py-8 text-text-muted">
              No orders found. Create your first order!
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

