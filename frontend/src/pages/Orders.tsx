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
    <div className="p-8">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 mb-2">Orders</h1>
            <p className="text-xl text-gray-600">Monitor and manage your trading orders</p>
          </div>
          <button
            onClick={() => setShowForm(!showForm)}
            disabled={tradingHalted}
            className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg shadow-lg hover:shadow-xl transition-all duration-300 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Plus size={20} />
            New Order
          </button>
        </div>

        {tradingHalted && (
          <div className="bg-red-50 border-2 border-red-200 rounded-2xl p-6 mb-8 flex items-center gap-4">
            <XCircle size={48} className="text-red-500" />
            <div>
              <h3 className="text-2xl font-bold text-red-900 mb-2">Emergency Stop Active</h3>
              <p className="text-lg text-red-800">Order creation and execution blocked</p>
            </div>
          </div>
        )}

        {showForm && (
          <div className="bg-white border-2 border-gray-200 rounded-3xl p-8 mb-12 shadow-2xl">
            <h2 className="text-3xl font-bold mb-8 text-gray-900 flex items-center gap-3">
              <Plus size={32} />
              Create New Order
            </h2>
            <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <label className="block text-lg font-semibold text-gray-700 mb-3">Symbol</label>
                <select 
                  value={newOrder.symbol} 
                  onChange={(e) => setNewOrder({ ...newOrder, symbol: e.target.value })} 
                  className="w-full p-4 border-2 border-gray-200 rounded-xl text-lg font-semibold focus:ring-4 focus:ring-blue-200 focus:border-blue-500 transition-all shadow-inner hover:shadow-md"
                >
                  <option value="BTCUSDT">BTCUSDT</option>
                  <option value="ETHUSDT">ETHUSDT</option>
                  <option value="SOLUSDT">SOLUSDT</option>
                  <option value="BNBUSDT">BNBUSDT</option>
                  <option value="EURUSD">EURUSD</option>
                </select>
              </div>
              <div>
                <label className="block text-lg font-semibold text-gray-700 mb-3">Side</label>
                <select 
                  value={newOrder.side} 
                  onChange={(e) => setNewOrder({ ...newOrder, side: e.target.value as 'BUY' | 'SELL' })} 
                  className="w-full p-4 border-2 border-gray-200 rounded-xl text-lg font-semibold focus:ring-4 focus:ring-blue-200 focus:border-blue-500 transition-all shadow-inner hover:shadow-md"
                >
                  <option value="BUY">BUY</option>
                  <option value="SELL">SELL</option>
                </select>
              </div>
              <div>
                <label className="block text-lg font-semibold text-gray-700 mb-3">Type</label>
                <select 
                  value={newOrder.order_type} 
                  onChange={(e) => setNewOrder({ ...newOrder, order_type: e.target.value as 'market' | 'limit' })} 
                  className="w-full p-4 border-2 border-gray-200 rounded-xl text-lg font-semibold focus:ring-4 focus:ring-blue-200 focus:border-blue-500 transition-all shadow-inner hover:shadow-md"
                >
                  <option value="market">Market</option>
                  <option value="limit">Limit</option>
                </select>
              </div>
              <div>
                <label className="block text-lg font-semibold text-gray-700 mb-3">Quantity</label>
                <input 
                  type="number" 
                  step="0.0001" 
                  value={newOrder.quantity} 
                  onChange={(e) => setNewOrder({ ...newOrder, quantity: parseFloat(e.target.value) })} 
                  className="w-full p-4 border-2 border-gray-200 rounded-xl text-lg font-semibold focus:ring-4 focus:ring-blue-200 focus:border-blue-500 transition-all shadow-inner hover:shadow-md"
                />
              </div>
              <div className="md:col-span-2 flex items-end gap-4">
                <button 
                  type="submit" 
                  disabled={createOrder.isPending || tradingHalted}
                  className="flex-1 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white font-bold py-4 px-8 rounded-xl shadow-2xl hover:shadow-3xl transition-all duration-300 transform hover:-translate-y-1 active:translate-y-0 disabled:opacity-50 disabled:cursor-not-allowed text-lg flex items-center gap-3"
                >
                  {createOrder.isPending && (
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" pathLength="1" className="opacity-25" />
                      <path d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" fill="currentColor" />
                    </svg>
                  )}
                  Create Order
                </button>
                <button 
                  type="button"
                  onClick={() => setShowForm(false)}
                  className="px-8 py-4 border-2 border-gray-300 hover:border-gray-400 text-gray-700 font-semibold rounded-xl transition-all duration-300 hover:shadow-lg text-lg hover:bg-gray-50"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        )}

        <div className="bg-white border-4 border-gray-100 rounded-3xl p-8 shadow-2xl">
          <h2 className="text-4xl font-bold text-gray-900 mb-12 flex items-center gap-4">
            <Clock size={48} className="text-blue-500" />
            Orders History
          </h2>
          
          <div className="overflow-x-auto rounded-2xl border-4 border-gray-200 shadow-inner">
            <table className="w-full">
              <thead>
                <tr className="bg-gradient-to-r from-gray-50 to-gray-100">
                  <th className="text-left py-6 px-8 text-lg font-bold text-gray-700 uppercase tracking-wide">Order ID</th>
                  <th className="text-left py-6 px-8 text-lg font-bold text-gray-700 uppercase tracking-wide">Symbol</th>
                  <th className="text-left py-6 px-8 text-lg font-bold text-gray-700 uppercase tracking-wide">Side</th>
                  <th className="text-left py-6 px-8 text-lg font-bold text-gray-700 uppercase tracking-wide">Type</th>
                  <th className="text-right py-6 px-8 text-lg font-bold text-gray-700 uppercase tracking-wide">Quantity</th>
                  <th className="text-right py-6 px-8 text-lg font-bold text-gray-700 uppercase tracking-wide">Avg Price</th>
                  <th className="text-center py-6 px-8 text-lg font-bold text-gray-700 uppercase tracking-wide">Status</th>
                  <th className="text-left py-6 px-8 text-lg font-bold text-gray-700 uppercase tracking-wide">Created</th>
                  <th className="text-center py-6 px-8 text-lg font-bold text-gray-700 uppercase tracking-wide w-40">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {ordersList.map((order: any, index: number) => (
                  <tr key={order.order_id || index} className="hover:bg-gray-50 transition-colors">
                    <td className="py-6 px-8 font-mono text-lg font-bold text-gray-900 border-r border-gray-200">
                      {order.order_id?.slice(0,8)}...
                    </td>
                    <td className="py-6 px-8 font-bold text-2xl text-blue-600">{order.symbol}</td>
                    <td className="py-6 px-8">
                      <span className={`px-6 py-3 rounded-full text-lg font-bold ${order.side === 'BUY' ? 'bg-green-100 text-green-800 border-2 border-green-300 shadow-lg' : 'bg-red-100 text-red-800 border-2 border-red-300 shadow-lg'}`}>
                        {order.side}
                      </span>
                    </td>
                    <td className="py-6 px-8 text-xl font-semibold text-gray-700 capitalize">{order.order_type}</td>
                    <td className="py-6 px-8 text-right font-mono text-xl font-bold text-gray-900">{order.quantity?.toFixed(6) || '0'}</td>
                    <td className="py-6 px-8 text-right font-mono text-xl font-bold text-gray-900">
                      {order.average_price ? `$${order.average_price.toFixed(2)}` : '-'}
                    </td>
                    <td className="py-6 px-8 text-center">
                      <div className={`flex items-center gap-3 p-4 rounded-2xl shadow-lg px-8 ${getStatusColor(order.status || '')} bg-opacity-20`}>
                        {getStatusIcon(order.status || 'pending')}
                        <span className="text-xl font-bold capitalize">
                          {order.status || 'unknown'}
                        </span>
                      </div>
                    </td>
                    <td className="py-6 px-8 text-lg text-gray-600">
                      {order.created_at ? new Date(order.created_at).toLocaleString() : 'N/A'}
                    </td>
                    <td className="py-6 px-8">
                      {order.status === 'pending' && (
                        <div className="flex gap-4 justify-center">
                          <button
                            onClick={() => executeOrder.mutate(order.order_id)}
                            disabled={tradingHalted}
                            className="p-4 bg-green-500 hover:bg-green-600 text-white rounded-2xl shadow-2xl hover:shadow-3xl transition-all duration-300 transform hover:scale-110 active:scale-100 flex items-center justify-center text-xl font-bold disabled:opacity-50"
                          >
                            <Play size={24} />
                          </button>
                          <button
                            onClick={() => cancelOrder.mutate(order.order_id)}
                            className="p-4 bg-red-500 hover:bg-red-600 text-white rounded-2xl shadow-2xl hover:shadow-3xl transition-all duration-300 transform hover:scale-110 active:scale-100 flex items-center justify-center text-xl font-bold"
                          >
                            <X size={24} />
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
                {ordersList.length === 0 && (
                  <tr>
                    <td colSpan={9} className="py-24 text-center">
                      <div className="flex flex-col items-center gap-8">
                        <Clock size={96} className="text-gray-300" />
                        <div className="space-y-4">
                          <h3 className="text-4xl font-bold text-gray-400">No Orders Yet</h3>
                          <p className="text-2xl text-gray-500">Your trading history starts when you place your first order</p>
                          <p className="text-lg text-gray-400">Orders automatically refresh every 10 seconds</p>
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

