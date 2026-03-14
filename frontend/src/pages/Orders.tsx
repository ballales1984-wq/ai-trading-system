import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { emergencyApi, ordersApi } from '../services/api';
import { Plus, Play, X, Clock, CheckCircle, XCircle } from 'lucide-react';

type OrderState = {
  symbol: string;
  side: 'BUY' | 'SELL';
  order_type: 'market' | 'limit';
  quantity: number;
};

export default function Orders() {
  const queryClient = useQueryClient();
  const [showForm, setShowForm] = useState(false);
  const [newOrder, setNewOrder] = useState<OrderState>({
    symbol: 'BTCUSDT',
    side: 'BUY',
    order_type: 'market',
    quantity: 0.001,
  });

  const { data: orders = [] } = useQuery({
    queryKey: ['orders'],
    queryFn: () => ordersApi.list(),
    refetchInterval: 10000,
  });

  const { data: emergencyStatus = { trading_halted: false } } = useQuery({
    queryKey: ['emergency-status'],
    queryFn: () => emergencyApi.getStatus(),
  });

  const tradingHalted = Boolean(emergencyStatus.trading_halted);

  const createOrder = useMutation({
    mutationFn: ordersApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['orders'] });
      setShowForm(false);
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
    createOrder.mutate({
      symbol: newOrder.symbol,
      side: newOrder.side,
      order_type: newOrder.order_type,
      quantity: newOrder.quantity
    });
  };

  const ordersList = Array.isArray(orders) ? orders : [];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'filled': 
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'pending': 
        return <Clock className="w-4 h-4 text-yellow-500" />;
      case 'cancelled': 
        return <XCircle className="w-4 h-4 text-red-500" />;
      default:
        return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'filled': return 'text-green-500 font-bold';
      case 'pending': return 'text-yellow-500 font-bold';
      case 'cancelled': return 'text-red-500 font-bold';
      default: return 'text-gray-500 font-bold';
    }
  };

  return (
    <div className="p-6">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text">Orders</h1>
          <p className="text-text-muted">Monitor and manage your trading orders</p>
        </div>
        <button
          onClick={() => setShowForm(!showForm)}
          disabled={tradingHalted}
          className="bg-primary hover:bg-primary/80 text-white font-bold py-2.5 px-6 rounded-xl shadow-lg glow-primary transition-all duration-300 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Plus size={20} />
          New Order
        </button>
      </div>

      {tradingHalted && (
        <div className="mb-6 bg-danger/10 border border-danger/30 rounded-xl p-4 flex items-center gap-3">
          <XCircle className="w-5 h-5 text-danger flex-shrink-0" />
          <div>
            <p className="text-danger font-bold text-sm uppercase tracking-wide">Emergency Stop Active</p>
            <p className="text-text-muted text-xs">Order creation and execution blocked by risk parameters.</p>
          </div>
        </div>
      )}

      {showForm && (
        <div className="premium-glass-panel p-6 mb-8 border-white/[0.05] animate-in fade-in slide-in-from-top-4 duration-300">
          <h2 className="text-lg font-bold mb-6 text-text flex items-center gap-2">
            <Plus size={20} className="text-primary" />
            Create New Order
          </h2>
          <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 items-end">
            <div className="space-y-2">
              <label className="text-xs font-bold text-text-muted uppercase tracking-wider pl-1">Symbol</label>
              <select 
                value={newOrder.symbol} 
                onChange={(e) => setNewOrder({ ...newOrder, symbol: e.target.value })} 
                className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-2.5 text-text font-medium focus:outline-none focus:border-primary/50"
              >
                <option value="BTCUSDT">BTCUSDT</option>
                <option value="ETHUSDT">ETHUSDT</option>
                <option value="SOLUSDT">SOLUSDT</option>
                <option value="BNBUSDT">BNBUSDT</option>
                <option value="EURUSD">EURUSD</option>
              </select>
            </div>
            <div className="space-y-2">
              <label className="text-xs font-bold text-text-muted uppercase tracking-wider pl-1">Side</label>
              <select 
                value={newOrder.side} 
                onChange={(e) => setNewOrder({ ...newOrder, side: e.target.value as 'BUY' | 'SELL' })} 
                className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-2.5 text-text font-medium focus:outline-none focus:border-primary/50"
              >
                <option value="BUY">BUY</option>
                <option value="SELL">SELL</option>
              </select>
            </div>
            <div className="space-y-2">
              <label className="text-xs font-bold text-text-muted uppercase tracking-wider pl-1">Quantity</label>
              <input 
                type="number" 
                step="0.0001" 
                value={newOrder.quantity} 
                onChange={(e) => setNewOrder({ ...newOrder, quantity: parseFloat(e.target.value) })} 
                className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-2.5 text-text font-medium focus:outline-none focus:border-primary/50"
              />
            </div>
            <div className="flex gap-2">
              <button 
                type="submit" 
                disabled={createOrder.isPending || tradingHalted}
                className="flex-1 bg-success/20 hover:bg-success/30 text-success border border-success/30 font-bold py-2.5 px-4 rounded-xl transition-all duration-300 disabled:opacity-50"
              >
                Place Order
              </button>
              <button 
                type="button"
                onClick={() => setShowForm(false)}
                className="px-4 py-2.5 bg-white/5 hover:bg-white/10 text-text-muted border border-white/10 rounded-xl transition-all"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="premium-glass-panel overflow-hidden border-white/[0.05]">
        <div className="px-6 py-5 border-b border-white/[0.05] bg-white/[0.02]">
          <h2 className="text-lg font-semibold text-text tracking-wide flex items-center gap-3">
            <Clock size={20} className="text-primary" />
            Orders History
          </h2>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-black/20">
              <tr>
                <th className="text-left py-4 px-6 text-[10px] text-text-muted font-bold uppercase tracking-widest">Symbol</th>
                <th className="text-left py-4 px-6 text-[10px] text-text-muted font-bold uppercase tracking-widest">Side</th>
                <th className="text-left py-4 px-6 text-[10px] text-text-muted font-bold uppercase tracking-widest">Type</th>
                <th className="text-right py-4 px-6 text-[10px] text-text-muted font-bold uppercase tracking-widest">Quantity</th>
                <th className="text-right py-4 px-6 text-[10px] text-text-muted font-bold uppercase tracking-widest">Avg Price</th>
                <th className="text-center py-4 px-6 text-[10px] text-text-muted font-bold uppercase tracking-widest">Status</th>
                <th className="text-left py-4 px-6 text-[10px] text-text-muted font-bold uppercase tracking-widest">Created</th>
                <th className="text-center py-4 px-6 text-[10px] text-text-muted font-bold uppercase tracking-widest">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.05]">
              {ordersList.map((order: any, index: number) => (
                <tr key={order.order_id || index} className="hover:bg-white/[0.02] transition-colors group">
                  <td className="py-4 px-6 font-bold text-text">{order.symbol}</td>
                  <td className="py-4 px-6">
                    <span className={`text-xs font-bold px-2 py-0.5 rounded border ${order.side === 'BUY' ? 'text-success border-success/30 bg-success/5' : 'text-danger border-danger/30 bg-danger/5'}`}>
                      {order.side}
                    </span>
                  </td>
                  <td className="py-4 px-6 text-sm text-text-muted capitalize">{order.order_type}</td>
                  <td className="py-4 px-6 text-right font-mono-num text-sm text-text font-medium">{order.quantity?.toFixed(6)}</td>
                  <td className="py-4 px-6 text-right font-mono-num text-sm text-text">
                    {order.average_price ? `$${order.average_price.toFixed(2)}` : '-'}
                  </td>
                  <td className="py-4 px-6 text-center">
                    <div className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider ${getStatusColor(order.status || '')} bg-opacity-10`}>
                      {getStatusIcon(order.status || 'pending')}
                      {order.status || 'unknown'}
                    </div>
                  </td>
                  <td className="py-4 px-6 text-xs text-text-muted">
                    {order.created_at ? new Date(order.created_at).toLocaleTimeString() : 'N/A'}
                  </td>
                  <td className="py-4 px-6">
                    {order.status === 'pending' && (
                      <div className="flex gap-2 justify-center">
                        <button
                          onClick={() => executeOrder.mutate(order.order_id)}
                          className="p-1.5 text-success hover:bg-success/20 rounded-lg transition-colors"
                          title="Execute"
                        >
                          <Play size={16} />
                        </button>
                        <button
                          onClick={() => cancelOrder.mutate(order.order_id)}
                          className="p-1.5 text-danger hover:bg-danger/20 rounded-lg transition-colors"
                          title="Cancel"
                        >
                          <X size={16} />
                        </button>
                      </div>
                    )}
                  </td>
                </tr>
              ))}
              {ordersList.length === 0 && (
                <tr>
                  <td colSpan={8} className="py-20 text-center">
                    <div className="flex flex-col items-center gap-3">
                      <Clock size={40} className="text-white/10" />
                      <div className="space-y-1">
                        <h3 className="text-text-muted font-bold">No Orders</h3>
                        <p className="text-xs text-text-muted/60 px-6">Active trading history will appear here.</p>
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
  );
}

