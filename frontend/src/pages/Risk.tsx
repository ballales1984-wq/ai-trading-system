import React from 'react';
import { Shield, Lock, AlertTriangle, Eye } from 'lucide-react';

export default function Risk() {
  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-text">Risk Management</h1>
        <p className="text-text-muted">Monitor exposure limits and safety parameters</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <RiskMetric title="Max Drawdown Limit" value="15.0%" status="Safe" />
        <RiskMetric title="Daily Loss Limit" value="2.0%" status="Safe" />
        <RiskMetric title="Max Leverage" value="10.0x" status="Warning" />
        <RiskMetric title="Auto-Stop Level" value="25.0%" status="Safe" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="premium-glass-panel p-6 border-white/[0.05]">
          <div className="flex items-center gap-3 mb-6">
            <Lock className="text-primary" size={20} />
            <h2 className="text-lg font-bold text-text">Risk Controls</h2>
          </div>
          <div className="space-y-4">
            <ControlSwitch label="Emergency Liquidation" active={false} description="Instantly close all positions if drawdown exceeds limits" />
            <ControlSwitch label="Correlated Exposure Guard" active={true} description="Prevents opening highly correlated positions automatically" />
            <ControlSwitch label="Volatility Circuit Breaker" active={true} description="Halts trading during extreme market volatility" />
          </div>
        </div>

        <div className="premium-glass-panel p-6 border-white/[0.05]">
          <div className="flex items-center gap-3 mb-6">
            <Eye className="text-primary" size={20} />
            <h2 className="text-lg font-bold text-text">Exposure Monitor</h2>
          </div>
          <div className="p-12 text-center text-text-muted">
            <AlertTriangle className="mx-auto mb-4 opacity-20" size={48} />
            <p className="text-sm">Real-time risk surface visualization is being initialized...</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function RiskMetric({ title, value, status }: any) {
  return (
    <div className="premium-glass-panel p-5 border-white/[0.05]">
      <div className="text-xs text-text-muted uppercase font-semibold mb-1">{title}</div>
      <div className="text-2xl font-bold font-mono-num text-text">{value}</div>
      <div className={`text-[10px] mt-2 font-bold uppercase tracking-tight inline-block px-1.5 py-0.5 rounded border 
        ${status === 'Safe' ? 'text-success border-success/30 bg-success/5' : 'text-yellow-500 border-yellow-500/30 bg-yellow-500/5'}`}>
        {status}
      </div>
    </div>
  );
}

function ControlSwitch({ label, active, description }: any) {
  return (
    <div className="flex items-center justify-between p-4 bg-white/[0.02] border border-white/[0.05] rounded-xl">
      <div>
        <div className="text-sm font-bold text-text">{label}</div>
        <div className="text-xs text-text-muted mt-0.5">{description}</div>
      </div>
      <div className={`w-10 h-5 rounded-full relative transition-colors ${active ? 'bg-primary' : 'bg-white/10'}`}>
        <div className={`absolute top-1 w-3 h-3 rounded-full bg-white transition-all ${active ? 'left-6' : 'left-1'}`} />
      </div>
    </div>
  );
}
