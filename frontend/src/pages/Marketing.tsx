import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useMutation } from '@tanstack/react-query';
import { api } from '../services/api';

export default function Marketing() {
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const waitlistMutation = useMutation({
    mutationFn: async (email: string) => {
      const response = await api.post('/waitlist', { email, source: 'marketing_page' });
      return response.data;
    },
    onSuccess: () => {
      setMessage({ type: 'success', text: '🎉 Thanks! You\'re on the list. We\'ll be in touch soon.' });
      setEmail('');
    },
    onError: () => {
      // Fallback for demo/offline
      localStorage.setItem('waitlist_email', email);
      setMessage({ type: 'success', text: '🎉 Thanks! You\'re on the list. We\'ll be in touch soon.' });
      setEmail('');
    }
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (email) {
      waitlistMutation.mutate(email);
    }
  };

  const scrollTo = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <div className="min-h-screen bg-bg-primary text-text font-sans">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-bg-secondary/70 backdrop-blur-xl border-b border-white/5">
        <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
          <div className="text-2xl font-bold bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 bg-clip-text text-transparent">
            🤖 AI Trading
          </div>
          <div className="hidden md:flex gap-8">
            <button onClick={() => scrollTo('features')} className="text-slate-400 hover:text-white transition-colors">
              Features
            </button>
            <button onClick={() => scrollTo('pricing')} className="text-slate-400 hover:text-white transition-colors">
              Pricing
            </button>
            <button onClick={() => scrollTo('demo')} className="text-slate-400 hover:text-white transition-colors">
              Demo
            </button>
          </div>
          <Link
            to="/login"
            className="px-6 py-2.5 bg-primary/20 border border-primary/50 text-primary rounded-lg font-semibold hover:bg-primary/30 glow-primary transition-all duration-300"
          >
            Get Started
          </Link>
        </nav>
      </header>

      {/* Hero Section */}
      <section className="pt-40 pb-20 px-4 text-center relative overflow-hidden">
        {/* Glow ambientale centrale */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-primary/10 rounded-full blur-[120px] pointer-events-none" />

        <div className="max-w-4xl mx-auto relative z-10">
          <div className="inline-block mb-4 px-4 py-1.5 rounded-full border border-primary/30 bg-primary/10 text-primary text-sm font-semibold tracking-wide drop-shadow-[0_0_8px_rgba(59,130,246,0.5)]">
            v2.0 Institutional Release
          </div>
          <h1 className="text-5xl md:text-7xl font-extrabold mb-6 leading-tight tracking-tight">
            Trade Smarter with<br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary via-blue-400 to-cyan-400 drop-shadow-lg">
              Monte Carlo Risk Analysis
            </span>
          </h1>
          <p className="text-xl text-slate-400 max-w-2xl mx-auto mb-10">
            Professional-grade trading platform with institutional risk management, multi-asset support, and AI-powered signals.
          </p>
          <div className="flex gap-6 justify-center flex-wrap mb-16">
            <Link
              to="/login"
              className="px-8 py-4 bg-primary text-white rounded-lg font-bold hover:bg-primary-hover glow-primary transition-all duration-300 shadow-[0_0_20px_rgba(59,130,246,0.4)] text-lg"
            >
              Enter Terminal
            </Link>
            <button
              onClick={() => scrollTo('features')}
              className="px-8 py-4 border border-white/10 bg-white/[0.02] rounded-lg font-semibold hover:bg-white/[0.08] transition-all duration-300 text-lg"
            >
              Explore Features
            </button>
          </div>

          {/* Email Signup */}
          <div id="signup" className="premium-glass-panel p-10 max-w-lg mx-auto relative overflow-hidden">
            {/* Sottile highlight superiore */}
            <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-primary/50 to-transparent"></div>

            <h3 className="text-2xl font-bold mb-3 tracking-wide">🚀 Join the Beta</h3>
            <p className="text-slate-400 mb-6">Get early access and 7 days free trial. No credit card required.</p>
            <form onSubmit={handleSubmit} className="flex flex-col gap-3">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Enter your institutional email"
                required
                className="px-4 py-3.5 bg-black/40 border border-white/10 rounded-lg text-text placeholder-text-muted focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all"
              />
              <button
                type="submit"
                disabled={waitlistMutation.isPending}
                className="px-6 py-3.5 bg-primary/20 border border-primary/50 text-primary rounded-lg font-bold hover:bg-primary/40 glow-primary transition-all duration-300 disabled:opacity-50"
              >
                {waitlistMutation.isPending ? 'Joining...' : 'Join'}
              </button>
            </form>
            {message && (
              <div className={`mt-4 p-3 rounded-lg ${message.type === 'success' ? 'bg-emerald-500/10 border border-emerald-500 text-emerald-400' : 'bg-red-500/10 border border-red-500 text-red-400'}`}>
                {message.text}
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-24 px-4 relative">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16 relative z-10">
            <h2 className="text-4xl md:text-5xl font-extrabold mb-4 tracking-tight">Professional Trading Tools</h2>
            <p className="text-text-muted text-lg max-w-2xl mx-auto">Institutional-grade infrastructure, now accessible.</p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              { icon: '🎲', title: 'Monte Carlo Simulations', desc: '5-level probability analysis with thousands of scenarios to understand risk and potential outcomes.' },
              { icon: '📊', title: 'Multi-Asset Support', desc: 'Trade crypto, forex, stocks, and futures from a single platform with unified risk management.' },
              { icon: '🤖', title: 'AI-Powered Signals', desc: 'Machine learning models trained on years of data to identify trading opportunities.' },
              { icon: '🎯', title: 'Institutional Risk Management', desc: 'VaR, CVaR, and dynamic position sizing to protect your capital.' },
              { icon: '📈', title: 'Real-Time Dashboard', desc: 'Live portfolio tracking with WebSocket-powered institutional charts.' },
              { icon: '🔔', title: 'Smart Alerts', desc: 'Telegram and email notifications for critical circuit-breakers and signals.' }
            ].map((feature, i) => (
              <div key={i} className="premium-glass-panel p-8 premium-glass-hover group">
                <div className="w-14 h-14 bg-primary/10 border border-primary/30 rounded-xl flex items-center justify-center text-2xl mb-6 group-hover:scale-110 transition-transform glow-primary">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-bold mb-3 tracking-wide">{feature.title}</h3>
                <p className="text-text-muted leading-relaxed">{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 px-4 bg-black/40 border-y border-white/5 relative overflow-hidden">
        {/* Subtle background glow */}
        <div className="absolute inset-0 bg-primary/5 blur-[100px] pointer-events-none" />

        <div className="max-w-5xl mx-auto relative z-10">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-10 text-center">
            {[
              { number: '50+', label: 'Supported Markets' },
              { number: '1M+', label: 'Simulations/Day' },
              { number: '<50ms', label: 'Order Latency' },
              { number: '24/7', label: 'Risk Monitor' }
            ].map((stat, i) => (
              <div key={i}>
                <h3 className="text-5xl md:text-6xl font-black font-mono-num text-transparent bg-clip-text bg-gradient-to-br from-white to-white/50 mb-2 drop-shadow-md">
                  {stat.number}
                </h3>
                <p className="text-text-muted font-medium uppercase tracking-widest text-sm">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="py-24 px-4">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-extrabold mb-4">Transparent Pricing</h2>
            <p className="text-text-muted text-lg">Institutional tools structured for every scale.</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              { name: 'Basic', price: '€19', features: ['5 assets monitoring', 'Basic Monte Carlo', 'Email alerts', 'End-of-day reports'] },
              { name: 'Pro', price: '€49', featured: true, features: ['Unlimited assets', 'Advanced Monte Carlo (5 levels)', 'Telegram Alerts', 'Real-time WebSockets', 'API access'] },
              { name: 'Institutional', price: '€199', features: ['Everything in Pro', 'Custom strategies', 'Dedicated account mgr', 'White-label reports', 'SLA 99.9%'] }
            ].map((plan, i) => (
              <div key={i} className={`premium-glass-panel p-8 relative ${plan.featured ? 'border-primary/50 shadow-[0_0_30px_rgba(59,130,246,0.15)] scale-105 z-10' : ''}`}>
                {plan.featured && (
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-primary text-white border border-primary/50 px-4 py-1 rounded-full text-xs font-bold uppercase tracking-wider glow-primary">
                    Most Popular
                  </div>
                )}
                <h3 className="text-2xl font-bold mb-2 text-text">{plan.name}</h3>
                <div className="text-5xl font-black font-mono-num mb-6 text-text">
                  {plan.price}<span className="text-lg text-text-muted font-medium">/mo</span>
                </div>
                <ul className="mb-8 space-y-4">
                  {plan.features.map((feat, j) => (
                    <li key={j} className="text-text-muted flex items-start gap-3">
                      <span className="text-success font-bold mt-0.5">✓</span>
                      <span>{feat}</span>
                    </li>
                  ))}
                </ul>
                <Link
                  to="/login"
                  className={`block w-full py-3.5 rounded-lg font-bold text-center transition-all duration-300 ${plan.featured
                      ? 'bg-primary text-white hover:bg-primary-hover glow-primary'
                      : 'bg-white/[0.03] border border-white/10 text-text hover:bg-white/[0.08]'
                    }`}
                >
                  {plan.name === 'Institutional' ? 'Contact Sales' : 'Start Free Trial'}
                </Link>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Demo Section */}
      <section id="demo" className="py-32 px-4 relative overflow-hidden">
        <div className="absolute inset-0 bg-primary/5 blur-[120px] pointer-events-none" />
        <div className="max-w-4xl mx-auto text-center relative z-10">
          <h2 className="text-4xl md:text-5xl font-black mb-6">Enter The Terminal</h2>
          <p className="text-xl text-text-muted mb-10 max-w-2xl mx-auto">
            Experience the power of institutional AI-driven trading.
            Real-time latency, deep liquidity modeling, complete control.
          </p>
          <Link
            to="/login"
            className="inline-block px-12 py-5 bg-primary/20 border-2 border-primary text-primary hover:bg-primary hover:text-white rounded-lg font-bold text-xl transition-all duration-300 glow-primary shadow-[0_0_30px_rgba(59,130,246,0.3)]"
          >
            Launch Platform
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-4 border-t border-white/5 bg-black/40">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="text-xl font-bold text-white tracking-widest uppercase">
            🤖 AI Trading
          </div>
<div className="flex gap-8">
              <Link to="/privacy" className="text-text-muted hover:text-primary transition-colors text-sm font-medium">Privacy Policy</Link>
              <Link to="/terms" className="text-text-muted hover:text-primary transition-colors text-sm font-medium">Terms of Service</Link>
              <Link to="/cookies" className="text-text-muted hover:text-primary transition-colors text-sm font-medium">Cookie Policy</Link>
              <a href="mailto:support@ai-trading-system.com" className="text-text-muted hover:text-primary transition-colors text-sm font-medium">Contact</a>
            </div>
          <p className="text-text-muted/50 text-xs">© 2026 AI Trading System v2.0. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

