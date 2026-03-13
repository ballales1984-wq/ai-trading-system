import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { emergencyApi, ordersApi } from '../services/api';
import { Plus, Play, X, Clock, CheckCircle, XCircle } from 'lucide-react';
import type { OrderCreate } from '../types';

export default function Orders() {
  const queryClient = useQueryClient();
  const [showForm, setShowForm] = useState(false);
  const [newOrder, setNewOrder] = useState<OrderCreate>({
    symbol: 'BTCUSDT',
    side: 'BUY',
    order_type: 'market',
    quantity: 0.001,
    broker: 'paper',
  });

  const { data: orders } = useQuery({
    queryKey: ['orders'],
    queryFn: () => ordersApi.list(),
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
        order_type: 'market',
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
    if (tradingHalted) return;
    createOrder.mutate(newOrder);
  };

  const ordersList = Array.isArray(orders) ? orders : [];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'filled': return <CheckCircle className="w-4 h-4 text-success" />;
      case 'pending': return <Clock className="w-4 h-4 text-warning" />;
      case 'cancelled': return <XCircle className="w-4 h-4 text-danger" />;
      default:
        return <Clock className="w-4 h-4 text-muted" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'filled': return 'text-success';
      case 'pending': return 'text-warning';
      case 'cancelled': return 'text-danger';
      default: return 'text-muted';
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Orders</h1>
          <p className="text-muted-foreground mt-1">Manage active and historical trading orders</p>
        </div>
        <button
          onClick={() => setShowForm(!showForm)}
          disabled={tradingHalted}
          className="flex items-center gap-2 px-6 py-2.5 bg-primary text-primary-foreground rounded-xl hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium shadow-sm hover:shadow-md"
        >
          <Plus className="w-4 h-4" />
          {showForm ? 'Cancel' : 'New Order'}
        </button>
      </div>

      {tradingHalted && (
        <div className="mb-6 p-4 rounded-2xl border-2 border-destructive/20 bg-destructive/5 text-destructive font-medium flex items-center gap-3">
          <div className="w-5 h-5 bg-destructive/20 rounded-lg flex items-center justify-center">
            <X className="w-3 h-3" />
          </div>
          Emergency Stop active: Order creation and execution blocked
        </div>
      )}

      {showForm && (
        <div className="bg-card border rounded-2xl p-8 mb-8 shadow-lg">
          <h2 className="text-2xl font-bold text-foreground mb-6">Create New Order</h2>
          <form onSubmit={handleSubmit} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-muted-foreground mb-2">Symbol</label>
              <select 
                value={newOrder.symbol} 
                onChange={(e) => setNewOrder({ ...newOrder, symbol: e.target.value })} 
                className="w-full px-4 py-3 bg-background border border-input rounded-xl text-foreground focus:ring-2 focus:ring-primary focus:border-primary transition-all"
              >
                <option value="BTCUSDT">BTC/USDT</option>
                <option value="ETHUSDT">ETH/USDT</option>
                <option value="SOLUSDT">SOL/USDT</option>
                <option value="BNBUSDT">BNB/USDT</option>
                <option value="EURUSD">EUR/USD</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-muted-foreground mb-2">Side</label>
              <select 
                value={newOrder.side} 
                onChange={(e) => setNewOrder({ ...newOrder, side: e.target.value })} 
                className="w-full px-4 py-3 bg-background border border-input rounded-xl text-foreground focus:ring-2 focus:ring-primary focus:border-primary transition-all"
              >
                <option value="BUY">BUY</option>
                <option value="SELL">SELL</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-muted-foreground mb-2">Type</label>
              <select 
                value={newOrder.order_type ?? ''} 
                onChange={(e) => setNewOrder({ ...newOrder, order_type: e.target.value })} 
                className="w-full px-4 py-3 bg-background border border-input rounded-xl text-foreground focus:ring-2 focus:ring-primary focus:border-primary transition-all"
              >
                <option value="market">Market</option>
                <option value="limit">Limit</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-muted-foreground mb-2">Quantity</label>
              <input 
                type="number" 
                step="0.0001" 
                min="0"
                value={newOrder.quantity} 
                onChange={(e) => setNewOrder({ ...newOrder, quantity: Number(e.target.value) })} 
                className="w-full px-4 py-3 bg-background border border-input rounded-xl text-foreground focus:ring-2 focus:ring-primary focus:border-primary transition-all"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-muted-foreground mb-2">Broker</label>
              <select 
                value={newOrder.broker ?? ''} 
                onChange={(e) => setNewOrder({ ...newOrder, broker: e.target.value })} 
                className="w-full px-4 py-3 bg-background border border-input rounded-xl text-foreground focus:ring-2 focus:ring-primary focus:border-primary transition-all"
              >
                <option value="paper">Paper Trading</option>
                <option value="binance">Binance</option>
                <option value="bybit">Bybit</option>
                <option value="ib">Interactive Brokers</option>
              </select>
            </div>
            <div className="flex flex-col sm:flex-row gap-3">
              <button 
                type="submit" 
                disabled={createOrder.isPending || tradingHalted}
                className="flex-1 px-6 py-3 bg-primary text-primary-foreground rounded-xl hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium shadow-lg hover:shadow-xl"
              >
                {createOrder.isPending ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
                      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" pathLength="0" className="opacity-25"/>
                      <path fill="currentColor" d="M4 12a8 8 0 0 1 8-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 0 1 4 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                    </svg>
                    Creating...
                  </>
                ) : (
                  'Create Order'
                )}
              </button>
              <button 
                type="button"
                onClick={() => setShowForm(false)}
                className="px-6 py-3 border border-input text-muted-foreground rounded-xl hover:bg-accent hover:text-foreground transition-all font-medium"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="space-y-6">
        <div className="bg-card border rounded-2xl p-6 shadow-lg">
          <h2 className="text-2xl font-bold text-foreground mb-6 flex items-center gap-3">
            Orders List
            <span className="text-sm text-muted-foreground font-normal">(Live updates every 10s)</span>
          </h2>
          <div className="overflow-x-auto rounded-xl border">
            <table className="w-full">
              <thead className="bg-muted/50">
                <tr>
                  <th className="text-left py-4 px-6 text-xs font-semibold text-muted-foreground uppercase tracking-wider">ID</th>
                  <th className="text-left py-4 px-6 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Symbol</th>
                  <th className="text-left py-4 px-6 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Side</th>
                  <th className="text-left py-4 px-6 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Type</th>
                  <th className="text-right py-4 px-6 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Qty</th>
                  <th className="text-right py-4 px-6 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Avg Price</th>
                  <th className="text-center py-4 px-6 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Status</th>
                  <th className="text-left py-4 px-6 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Created</th>
                  <th className="text-center py-4 px-6 text-xs font-semibold text-muted-foreground uppercase tracking-wider w-28">Actions</th>
                </tr>
              </thead>
              <tbody>
                {ordersList.map((order: any) => (
                  <tr key={order.order_id} className="border-t border-border hover:bg-accent transition-colors">
                    <td className="py-4 px-6 font-mono text-sm text-muted-foreground">
                      {order.order_id.slice(0,8)}...
                    </td>
                    <td className="py-4 px-6 font-semibold text-foreground">{order.symbol}</td>
                    <td className={`py-4 px-6 font-semibold px-2 py-1 rounded-full text-xs ${order.side === 'BUY' ? 'bg-success/10 text-success' : 'bg-destructive/10 text-destructive'}`}>
                      {order.side}
                    </td>
                    <td className="py-4 px-6 text-sm text-muted-foreground capitalize">{order.order_type}</td>
                    <td className="py-4 px-6 text-right font-mono text-sm">{order.quantity.toFixed(6)}</td>
                    <td className="py-4 px-6 text-right font-mono text-foreground/80">
                      {order.average_price ? `$${order.average_price.toFixed(2)}` : '-'}
                    </td>
                    <td className="py-4 px-6 text-center">
                      <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-muted/50">
                        {getStatusIcon(order.status)}
                        <span className={`${getStatusColor(order.status)} font-semibold capitalize text-xs`}>
                          {order.status}
                        </span>
                      </div>
                    </td>
                    <td className="py-4 px-6 text-xs text-muted-foreground">
                      {new Date(order.created_at).toLocaleString('en-US', { 
                        year: 'numeric', 
                        month: 'short', 
                        day: 'numeric', 
                        hour: '2-digit', 
                        minute: '2-digit' 
                      })}
                    </td>
                    <td className="py-4 px-6">
                      {order.status === 'pending' && (
                        <div className="flex items-center justify-center gap-2">
                          <button
                            onClick={() => executeOrder.mutate(order.order_id)}
                            disabled={tradingHalted || executeOrder.isPending}
                            className="p-2 bg-success/10 text-success hover:bg-success/20 border border-success/20 rounded-lg transition-all flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed shadow-sm hover:shadow-md"
                            title="Execute Order"
                          >
                            <Play className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => cancelOrder.mutate(order.order_id)}
                            disabled={cancelOrder.isPending}
                            className="p-2 bg-destructive/10 text-destructive hover:bg-destructive/20 border border-destructive/20 rounded-lg transition-all flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed shadow-sm hover:shadow-md"
                            title="Cancel Order"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
                {ordersList.length === 0 && (
                  <tr>
                    <td colSpan={9} className="py-20 text-center text-muted-foreground">
                      <div className="flex flex-col items-center gap-4">
                        <Clock className="w-16 h-16 opacity-40" />
                        <div>
                          <h3 className="text-xl font-semibold mb-1">No Orders</h3>
                          <p className="text-sm">Your order history is empty. Create your first order!</p>
                        </div>
                      </div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

