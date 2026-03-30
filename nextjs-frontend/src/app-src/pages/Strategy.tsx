import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { strategyApi } from '../services/api';
import { TrendingUp, Activity, Target, Zap, Play, Loader2 } from 'lucide-react';

export default function Strategy() {
  const queryClient = useQueryClient();

  const { data: strategies, isLoading, error } = useQuery({
    queryKey: ['strategies'],
    queryFn: () => strategyApi.list(),
  });

  const runMutation = useMutation({
    mutationFn: (id: string) => strategyApi.run(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
    }
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 className="w-8 h-8 text-primary animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 text-center">
        <p className="text-danger font-bold">Error loading strategies</p>
        <p className="text-text-muted text-sm mt-2">Make sure the backend is running.</p>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-text">Trading Strategies</h1>
        <p className="text-text-muted">Manage and monitor automated trading algorithms</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {strategies?.map((strat) => (
          <StrategyCard
            key={strat.strategy_id}
            name={strat.name}
            status={strat.enabled ? 'Active' : 'Paused'}
            pnl="+0.0%" // Performance metrics could be fetched separately if needed
            description={strat.description}
            type={strat.strategy_type}
            onRun={() => runMutation.mutate(strat.strategy_id)}
            isRunning={runMutation.isPending && runMutation.variables === strat.strategy_id}
          />
        ))}
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

function StrategyCard({ name, status, pnl, description, type, onRun, isRunning }: any) {
  const getIcon = (type: string) => {
    switch (type) {
      case 'momentum': return Zap;
      case 'mean_reversion': return Activity;
      case 'ml': return TrendingUp;
      default: return Target;
    }
  };

  const getColor = (type: string) => {
    switch (type) {
      case 'momentum': return 'text-primary border-primary/30 bg-primary/10';
      case 'mean_reversion': return 'text-yellow-500 border-yellow-500/30 bg-yellow-500/10';
      case 'ml': return 'text-purple border-purple/30 bg-purple/10';
      default: return 'text-text-muted border-white/10 bg-white/5';
    }
  };

  const Icon = getIcon(type);
  const colorClass = getColor(type);

  return (
    <div className="premium-glass-panel p-6 premium-glass-hover border-white/[0.05]">
      <div className="flex items-center justify-between mb-4">
        <div className={`p-3 rounded-xl border ${colorClass}`}>
          <Icon size={24} />
        </div>
        <div className="flex items-center gap-2">
          {status === 'Active' && (
            <button
              onClick={onRun}
              disabled={isRunning}
              className="p-1.5 rounded-lg bg-success/10 text-success border border-success/30 hover:bg-success/20 transition-colors"
              title="Run Strategy"
            >
              {isRunning ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
            </button>
          )}
          <span className={`text-xs font-bold px-2 py-1 rounded-full uppercase tracking-wider 
            ${status === 'Active' ? 'bg-success/10 text-success border border-success/30' : 'bg-white/5 text-text-muted border border-white/10'}`}>
            {status}
          </span>
        </div>
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
