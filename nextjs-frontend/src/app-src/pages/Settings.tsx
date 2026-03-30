import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { healthApi, cacheApi } from '../services/api';
import { User, Key, Bell, Globe, Activity, Database, Trash2, Loader2, CheckCircle2 } from 'lucide-react';

export default function Settings() {
  const queryClient = useQueryClient();

  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: healthApi.getStatus,
    refetchInterval: 30000,
  });

  const { data: cacheStats } = useQuery({
    queryKey: ['cache-stats'],
    queryFn: cacheApi.getStats,
    refetchInterval: 30000,
  });

  const clearCacheMutation = useMutation({
    mutationFn: cacheApi.clearAll,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cache-stats'] });
    }
  });

  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-text">Settings</h1>
        <p className="text-text-muted">Configuration and security preferences</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1 space-y-4">
          <SettingsNav icon={User} label="Profile" active />
          <SettingsNav icon={Key} label="API Keys" />
          <SettingsNav icon={Bell} label="Notifications" />
          <SettingsNav icon={Globe} label="Regional" />
          <SettingsNav icon={Activity} label="System Status" />
        </div>

        <div className="lg:col-span-2 space-y-6">
          {/* System Health */}
          <div className="premium-glass-panel p-6 border-white/[0.05]">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-bold text-text">System Health</h2>
              {healthLoading ? <Loader2 size={18} className="animate-spin text-primary" /> : (
                <div className="flex items-center gap-2">
                  <span className={`w-2.5 h-2.5 rounded-full ${health?.status === 'healthy' ? 'bg-success animate-pulse' : 'bg-danger'}`} />
                  <span className="text-xs font-bold uppercase tracking-wider">{health?.status || 'Unknown'}</span>
                </div>
              )}
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <HealthItem label="Environment" value={health?.environment || '...'} />
              <HealthItem label="App Version" value={health?.version || '...'} />
              <HealthItem label="Server Time" value={health ? new Date(health.timestamp).toLocaleTimeString() : '...'} />
            </div>
          </div>

          {/* Cache Management */}
          <div className="premium-glass-panel p-6 border-white/[0.05]">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <Database className="text-primary" size={20} />
                <h2 className="text-lg font-bold text-text">Cache Infrastructure</h2>
              </div>
              <button 
                onClick={() => clearCacheMutation.mutate()}
                disabled={clearCacheMutation.isPending}
                className="flex items-center gap-2 px-3 py-1.5 bg-danger/10 text-danger border border-danger/30 rounded-lg hover:bg-danger/20 transition-all text-xs font-bold"
              >
                {clearCacheMutation.isPending ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />}
                Clear All
              </button>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-white/[0.02] border border-white/[0.05] rounded-xl">
                <div>
                  <div className="text-sm font-bold text-text">In-Memory Cache</div>
                  <div className="text-xs text-text-muted mt-1">
                    {cacheStats?.in_memory.size} entries | {((cacheStats?.in_memory.hit_rate || 0) * 100).toFixed(1)}% hit rate
                  </div>
                </div>
                {cacheStats?.in_memory.available ? <CheckCircle2 size={20} className="text-success" /> : <Activity size={20} className="text-text-muted opacity-20" />}
              </div>

              <div className="flex items-center justify-between p-4 bg-white/[0.02] border border-white/[0.05] rounded-xl">
                <div>
                  <div className="text-sm font-bold text-text">Redis Distribution</div>
                  <div className="text-xs text-text-muted mt-1">
                    {cacheStats?.redis.keys} keys | {cacheStats?.redis.memory} used
                  </div>
                </div>
                {cacheStats?.redis.connected ? <CheckCircle2 size={20} className="text-success" /> : <Activity size={20} className="text-danger" />}
              </div>
            </div>
          </div>

          {/* Original Settings Content */}
          <div className="premium-glass-panel p-6 border-white/[0.05]">
            <h2 className="text-lg font-bold text-text mb-6">Broker Connectivity</h2>
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <InputGroup label="Binance API Key" placeholder="••••••••••••••••" />
                <InputGroup label="Binance Secret" placeholder="••••••••••••••••" />
              </div>
              <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-xl flex gap-3">
                <Bell className="text-primary shrink-0" size={20} />
                <p className="text-sm text-text-muted leading-relaxed">
                  API keys are encrypted and stored locally. Never share your secret keys with anyone.
                </p>
              </div>
              <button className="px-6 py-2.5 bg-primary text-white rounded-lg font-semibold hover:bg-primary/80 transition-colors">
                Save Configuration
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function HealthItem({ label, value }: { label: string, value: string }) {
  return (
    <div className="p-3 bg-white/[0.02] border border-white/[0.05] rounded-xl">
      <div className="text-[10px] text-text-muted uppercase font-bold mb-1 tracking-wider">{label}</div>
      <div className="text-sm font-bold text-text">{value}</div>
    </div>
  );
}

function SettingsNav({ icon: Icon, label, active }: any) {
  return (
    <div className={`flex items-center gap-3 px-4 py-3 rounded-xl cursor-pointer transition-all 
      ${active ? 'bg-primary/20 text-primary border border-primary/30' : 'bg-white/5 text-text-muted hover:text-text hover:bg-white/10 border border-transparent'}`}>
      <Icon size={18} />
      <span className="font-semibold text-sm">{label}</span>
    </div>
  );
}

function InputGroup({ label, placeholder }: any) {
  return (
    <div className="space-y-2">
      <label className="text-xs font-bold text-text-muted uppercase tracking-wider pl-1">{label}</label>
      <input 
        type="password" 
        readOnly
        placeholder={placeholder}
        className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-primary/50"
      />
    </div>
  );
}
