import React from 'react';
import { TrendingUp, Activity, Target, Zap } from 'lucide-react';

export default function Strategy() {
  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-text">Trading Strategies</h1>
        <p className="text-text-muted">Manage and monitor automated trading algorithms</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <StrategyCard 
          name="Momentum Alpha" 
          status="Active" 
          pnl="+12.4%" 
          description="High-frequency momentum strategy tracking top 10 liquid assets."
          icon={Zap}
          color="primary"
        />
        <StrategyCard 
          name="Mean Reversion" 
          status="Paused" 
          pnl="-2.1%" 
          description="Contrarian strategy focusing on RSI extremes and Bollinger Band breakouts."
          icon={Activity}
          color="yellow"
        />
        <StrategyCard 
          name="Neural Trend" 
          status="Testing" 
          pnl="+0.0%" 
          description="Deep learning model predicting trend continuations using multi-head attention."
          icon={TrendingUp}
          color="purple"
        />
      </div>

      <div className="mt-12 premium-glass-panel p-12 text-center">
        <div className="max-w-md mx-auto">
          <div className="w-16 h-16 bg-primary/10 rounded-2xl flex items-center justify-center mx-auto mb-6 border border-primary/20">
            <Target className="text-primary w-8 h-8" />
          </div>
          <h2 className="text-xl font-bold text-text mb-4">Strategy Builder Coming Soon</h2>
          <p className="text-text-muted mb-8">
            You will soon be able to create, backtest, and deploy your own custom strategies using our visual node editor or Python SDK.
          </p>
          <button className="px-6 py-3 bg-primary text-white rounded-xl font-semibold opacity-50 cursor-not-allowed">
            Create Strategy
          </button>
        </div>
      </div>
    </div>
  );
}

function StrategyCard({ name, status, pnl, description, icon: Icon, color }: any) {
  const colorMap: any = {
    primary: 'text-primary border-primary/30 bg-primary/10',
    yellow: 'text-yellow-500 border-yellow-500/30 bg-yellow-500/10',
    purple: 'text-purple border-purple/30 bg-purple/10',
  };

  return (
    <div className="premium-glass-panel p-6 premium-glass-hover border-white/[0.05]">
      <div className="flex items-center justify-between mb-4">
        <div className={`p-3 rounded-xl border ${colorMap[color]}`}>
          <Icon size={24} />
        </div>
        <span className={`text-xs font-bold px-2 py-1 rounded-full uppercase tracking-wider 
          ${status === 'Active' ? 'bg-success/10 text-success border border-success/30' : 'bg-white/5 text-text-muted border border-white/10'}`}>
          {status}
        </span>
      </div>
      <h3 className="text-lg font-bold text-text mb-2">{name}</h3>
      <p className="text-sm text-text-muted mb-6 line-clamp-2">{description}</p>
      <div className="flex items-center justify-between pt-4 border-t border-white/[0.05]">
        <span className="text-xs text-text-muted uppercase font-semibold">Total P&L</span>
        <span className={`font-mono-num font-bold ${pnl.startsWith('+') ? 'text-success' : 'text-danger'}`}>{pnl}</span>
      </div>
    </div>
  );
}
