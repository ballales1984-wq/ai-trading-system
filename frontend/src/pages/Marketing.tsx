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
    <div className="min-h-screen bg-slate-900 text-white">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-slate-900/90 backdrop-blur-md border-b border-white/10">
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
            className="px-6 py-2.5 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-lg font-semibold hover:opacity-90 transition-opacity"
          >
            Get Started
          </Link>
        </nav>
      </header>

      {/* Hero Section */}
      <section className="pt-40 pb-20 px-4 text-center relative overflow-hidden">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-indigo-500/10 rounded-full pointer-events-none" />
        <div className="max-w-4xl mx-auto">
          <h1 className="text-5xl md:text-6xl font-bold mb-6 leading-tight">
            Trade Smarter with<br />
            <span className="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 bg-clip-text text-transparent">
              Monte Carlo Risk Analysis
            </span>
          </h1>
          <p className="text-xl text-slate-400 max-w-2xl mx-auto mb-10">
            Professional-grade trading platform with institutional risk management, multi-asset support, and AI-powered signals.
          </p>
          <div className="flex gap-4 justify-center flex-wrap mb-16">
            <Link
              to="/login"
              className="px-8 py-4 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-lg font-semibold hover:opacity-90 transition-opacity text-lg"
            >
              View Demo
            </Link>
            <button
              onClick={() => scrollTo('features')}
              className="px-8 py-4 border border-white/20 rounded-lg font-semibold hover:bg-white/10 transition-colors text-lg"
            >
              Learn More
            </button>
          </div>

          {/* Email Signup */}
          <div id="signup" className="bg-slate-800 p-10 rounded-2xl max-w-lg mx-auto border border-white/10">
            <h3 className="text-2xl font-semibold mb-3">🚀 Join the Beta</h3>
            <p className="text-slate-400 mb-6">Get early access and 7 days free trial. No credit card required.</p>
            <form onSubmit={handleSubmit} className="flex flex-col gap-3">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Enter your email"
                required
                className="px-4 py-3.5 bg-slate-900 border border-white/10 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500"
              />
              <button
                type="submit"
                disabled={waitlistMutation.isPending}
                className="px-6 py-3.5 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-lg font-semibold hover:opacity-90 transition-opacity disabled:opacity-50"
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
      <section id="features" className="py-24 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Professional Trading Tools</h2>
            <p className="text-slate-400 max-w-2xl mx-auto">Everything you need for informed trading decisions</p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              { icon: '🎲', title: 'Monte Carlo Simulations', desc: '5-level probability analysis with thousands of scenarios to understand risk and potential outcomes.' },
              { icon: '📊', title: 'Multi-Asset Support', desc: 'Trade crypto, forex, stocks, and futures from a single platform with unified risk management.' },
              { icon: '🤖', title: 'AI-Powered Signals', desc: 'Machine learning models trained on years of data to identify trading opportunities.' },
              { icon: '🛡️', title: 'Institutional Risk Management', desc: 'VaR, CVaR, and dynamic position sizing to protect your capital.' },
              { icon: '📈', title: 'Real-Time Dashboard', desc: 'Live portfolio tracking with interactive charts and performance metrics.' },
              { icon: '🔔', title: 'Smart Alerts', desc: 'Telegram and email notifications for important market events and trade signals.' }
            ].map((feature, i) => (
              <div key={i} className="bg-slate-800 p-8 rounded-2xl border border-white/5 hover:border-indigo-500/30 transition-all hover:-translate-y-1">
                <div className="w-14 h-14 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-xl flex items-center justify-center text-2xl mb-5">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
                <p className="text-slate-400">{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 px-4 bg-slate-800">
        <div className="max-w-4xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-10 text-center">
            {[
              { number: '50+', label: 'Supported Assets' },
              { number: '10K+', label: 'Simulations/Day' },
              { number: '99.9%', label: 'Uptime' },
              { number: '24/7', label: 'Monitoring' }
            ].map((stat, i) => (
              <div key={i}>
                <h3 className="text-5xl font-bold bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 bg-clip-text text-transparent mb-2">
                  {stat.number}
                </h3>
                <p className="text-slate-400">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="py-24 px-4">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Simple, Transparent Pricing</h2>
            <p className="text-slate-400">Start with a 7-day free trial. No credit card required.</p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8">
            {[
              { name: 'Basic', price: '€19', features: ['5 assets monitoring', 'Basic Monte Carlo', 'Email alerts', 'Daily reports', 'Community support'] },
              { name: 'Pro', price: '€49', featured: true, features: ['Unlimited assets', 'Advanced Monte Carlo (5 levels)', 'Telegram + Email alerts', 'Real-time dashboard', 'API access', 'Priority support'] },
              { name: 'Enterprise', price: '€199', features: ['Everything in Pro', 'Custom strategies', 'Dedicated support', 'White-label options', 'SLA guarantee'] }
            ].map((plan, i) => (
              <div key={i} className={`bg-slate-800 p-8 rounded-2xl border ${plan.featured ? 'border-indigo-500 relative scale-105' : 'border-white/5'} ${plan.featured ? '' : ''}`}>
                {plan.featured && (
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 px-4 py-1 rounded-full text-sm font-semibold">
                    Most Popular
                  </div>
                )}
                <h3 className="text-2xl font-semibold mb-2">{plan.name}</h3>
                <div className="text-4xl font-bold mb-4">{plan.price}<span className="text-lg text-slate-400 font-normal">/month</span></div>
                <ul className="mb-8">
                  {plan.features.map((feat, j) => (
                    <li key={j} className="py-2.5 text-slate-400 flex items-center gap-2.5">
                      <span className="text-emerald-400 font-bold">✓</span> {feat}
                    </li>
                  ))}
                </ul>
                <Link
                  to="/login"
                  className={`block w-full py-3 rounded-lg font-semibold text-center transition-colors ${plan.featured ? 'bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 hover:opacity-90' : 'border border-white/20 hover:bg-white/10'}`}
                >
                  {plan.name === 'Enterprise' ? 'Contact Sales' : 'Start Free Trial'}
                </Link>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Demo Section */}
      <section id="demo" className="py-24 px-4 bg-slate-800">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl font-bold mb-6">Try It Now</h2>
          <p className="text-xl text-slate-400 mb-10">
            Experience the power of AI-driven trading with our interactive demo.
          </p>
          <Link
            to="/login"
            className="inline-block px-10 py-5 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-lg font-semibold text-xl hover:opacity-90 transition-opacity"
          >
            Launch Demo →
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-4 border-t border-white/10">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="text-2xl font-bold bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 bg-clip-text text-transparent">
            🤖 AI Trading
          </div>
          <div className="flex gap-8">
            <a href="#" className="text-slate-400 hover:text-white transition-colors">Privacy Policy</a>
            <a href="#" className="text-slate-400 hover:text-white transition-colors">Terms of Service</a>
            <a href="#" className="text-slate-400 hover:text-white transition-colors">Contact</a>
          </div>
          <p className="text-slate-500 text-sm">© 2024 AI Trading System. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

