import { useQuery } from '@tanstack/react-query';
import { riskApi } from '../services/api';
import { Shield, Lock, Loader2, BarChart3 } from 'lucide-react';

export default function Risk() {
  const { data: metrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['risk-metrics'],
    queryFn: () => riskApi.getMetrics(),
  });

  const { data: limits, isLoading: limitsLoading } = useQuery({
    queryKey: ['risk-limits'],
    queryFn: () => riskApi.getLimits(),
  });

  const { data: positions, isLoading: positionsLoading } = useQuery({
    queryKey: ['risk-positions'],
    queryFn: () => riskApi.getPositionRisks(),
  });

  const { data: controls, isLoading: controlsLoading } = useQuery({
    queryKey: ['risk-controls'],
    queryFn: () => riskApi.getControls(),
  });

  if (metricsLoading || limitsLoading || positionsLoading || controlsLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 className="w-8 h-8 text-primary animate-spin" />
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-text">Risk Management</h1>
        <p className="text-text-muted">Monitor exposure limits and safety parameters</p>
      </div>

      {/* Main Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <RiskMetric title="Current Leverage" value={`${metrics?.leverage.toFixed(1)}x`} status={metrics && metrics.leverage > 2 ? 'Warning' : 'Safe'} />
        <RiskMetric title="Margin Utilization" value={`${((metrics?.margin_utilization || 0) * 100).toFixed(1)}%`} status={metrics && metrics.margin_utilization > 0.05 ? 'Danger' : 'Safe'} />
        <RiskMetric title="VaR (1d, 95%)" value={`$${metrics?.var_1d.toLocaleString()}`} status="Safe" />
        <RiskMetric title="Volatility (Ann)" value={`${((metrics?.volatility || 0) * 100).toFixed(1)}%`} status="Safe" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Risk Limits */}
        <div className="premium-glass-panel p-6 border-white/[0.05]">
          <div className="flex items-center gap-3 mb-6">
            <Shield className="text-primary" size={20} />
            <h2 className="text-lg font-bold text-text">Institutional Limits</h2>
          </div>
          <div className="space-y-4">
            {limits?.map((limit) => (
              <div key={limit.limit_id} className="p-4 bg-white/[0.02] border border-white/[0.05] rounded-xl">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-bold text-text uppercase">{limit.limit_type.replace('_', ' ')}</span>
                  <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded border ${limit.severity === 'green' ? 'text-success border-success/30' : 'text-yellow-500 border-yellow-500/30'
                    }`}>
                    {limit.used_percentage}% USED
                  </span>
                </div>
                <div className="h-2 bg-black/40 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${limit.is_breached ? 'bg-danger' : limit.severity === 'yellow' ? 'bg-yellow-500' : 'bg-primary'}`}
                    style={{ width: `${limit.used_percentage}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Risk Controls */}
        <div className="premium-glass-panel p-6 border-white/[0.05]">
          <div className="flex items-center gap-3 mb-6">
            <Lock className="text-primary" size={20} />
            <h2 className="text-lg font-bold text-text">Risk Controls</h2>
          </div>
          <div className="space-y-4">
            {controls?.map((control) => (
              <ControlSwitch
                key={control.id}
                label={control.name}
                active={control.status === 'ACTIVE'}
                description={control.description}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Exposure Monitor */}
      <div className="premium-glass-panel p-6 border-white/[0.05]">
        <div className="flex items-center gap-3 mb-6">
          <BarChart3 className="text-primary" size={20} />
          <h2 className="text-lg font-bold text-text">Asset Concentration Risk</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-xs text-text-muted uppercase tracking-wider border-b border-white/5">
                <th className="pb-3 px-2">Symbol</th>
                <th className="pb-3 px-2 text-right">Market Value</th>
                <th className="pb-3 px-2 text-right">VaR Contribution</th>
                <th className="pb-3 px-2 text-right">Concentration</th>
                <th className="pb-3 px-2 text-right">Beta Exp</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {positions?.map((pos) => (
                <tr key={pos.symbol} className="text-sm hover:bg-white/[0.02]">
                  <td className="py-3 px-2 font-bold">{pos.symbol}</td>
                  <td className="py-3 px-2 text-right font-mono-num">${pos.market_value.toLocaleString()}</td>
                  <td className="py-3 px-2 text-right font-mono-num">${pos.var_contribution.toLocaleString()}</td>
                  <td className="py-3 px-2 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <span className="font-mono-num">{(pos.concentration_risk * 100).toFixed(0)}%</span>
                      <div className="w-16 h-1.5 bg-black/40 rounded-full overflow-hidden">
                        <div className="h-full bg-primary" style={{ width: `${pos.concentration_risk * 100}%` }} />
                      </div>
                    </div>
                  </td>
                  <td className="py-3 px-2 text-right font-mono-num">{pos.beta_weighted_exposure.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function RiskMetric({ title, value, status }: any) {
  const statusColor = status === 'Danger' ? 'text-danger border-danger/30 bg-danger/5' :
    status === 'Warning' ? 'text-yellow-500 border-yellow-500/30 bg-yellow-500/5' :
      'text-success border-success/30 bg-success/5';

  return (
    <div className="premium-glass-panel p-5 border-white/[0.05]">
      <div className="text-xs text-text-muted uppercase font-semibold mb-1">{title}</div>
      <div className="text-2xl font-bold font-mono-num text-text">{value}</div>
      <div className={`text-[10px] mt-2 font-bold uppercase tracking-tight inline-block px-1.5 py-0.5 rounded border ${statusColor}`}>
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
