import React from 'react';
import { Settings as SettingsIcon, User, Key, Bell, Globe } from 'lucide-react';

export default function Settings() {
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
        </div>

        <div className="lg:col-span-2 space-y-6">
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

          <div className="premium-glass-panel p-6 border-white/[0.05]">
            <h2 className="text-lg font-bold text-text mb-6">User Preferences</h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between text-sm py-2">
                <span className="text-text">Compact UI Mode</span>
                <span className="text-text-muted">Enabled</span>
              </div>
              <div className="flex items-center justify-between text-sm py-2 border-t border-white/[0.05]">
                <span className="text-text">Sound Notifications</span>
                <span className="text-text-muted">Disabled</span>
              </div>
              <div className="flex items-center justify-between text-sm py-2 border-t border-white/[0.05]">
                <span className="text-text">Default Currency</span>
                <span className="text-text-muted">USD</span>
              </div>
            </div>
          </div>
        </div>
      </div>
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
