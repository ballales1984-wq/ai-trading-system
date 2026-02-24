import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { useMutation, useQuery } from '@tanstack/react-query';
import { ArrowRight, BadgeCheck, ShieldAlert, Zap } from 'lucide-react';
import { waitlistApi } from '../services/api';

function parseError(error: unknown): string {
  if (
    typeof error === 'object' &&
    error &&
    'response' in error &&
    typeof (error as { response?: { data?: { detail?: string } } }).response?.data?.detail === 'string'
  ) {
    return (error as { response: { data: { detail: string } } }).response.data.detail;
  }
  return 'Unable to join waitlist right now. Please try again.';
}

export default function Marketing() {
  const [email, setEmail] = useState('');
  const [source, setSource] = useState('landing_react');

  const { data: waitlistCount } = useQuery({
    queryKey: ['waitlist-count'],
    queryFn: waitlistApi.count,
    retry: 1,
    staleTime: 60000,
  });

  const joinWaitlist = useMutation({
    mutationFn: waitlistApi.join,
    onSuccess: () => setEmail(''),
  });

  const signupMessage = useMemo(() => {
    if (joinWaitlist.isSuccess && joinWaitlist.data?.success) {
      const positionText =
        typeof joinWaitlist.data.position === 'number'
          ? ` You are currently #${joinWaitlist.data.position}.`
          : '';
      return `${joinWaitlist.data.message}${positionText}`;
    }
    if (joinWaitlist.isError) {
      return parseError(joinWaitlist.error);
    }
    return '';
  }, [joinWaitlist.data, joinWaitlist.error, joinWaitlist.isError, joinWaitlist.isSuccess]);

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    joinWaitlist.mutate({
      email: email.trim(),
      source,
    });
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(40,157,143,0.2),_transparent_42%),linear-gradient(170deg,_#0a131b,_#09131d_55%,_#111018)] text-text">
      <header className="border-b border-border/70 backdrop-blur-sm">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="font-['Sora'] text-xl font-bold tracking-tight">AI Trading System</div>
          <div className="flex items-center gap-3">
            <Link to="/legal/risk" className="text-sm text-text-muted hover:text-text">
              Risk Disclosure
            </Link>
            <Link to="/dashboard" className="rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-white hover:bg-primary/80">
              Dashboard
            </Link>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-16">
        <section className="grid gap-10 lg:grid-cols-2 lg:items-start">
          <div>
            <p className="mb-4 inline-flex items-center rounded-full border border-border bg-surface px-3 py-1 text-xs text-text-muted">
              EU-ready trading operations stack
            </p>
            <h1 className="font-['Sora'] text-4xl font-extrabold leading-tight md:text-5xl">
              Build with data, trade with controls.
            </h1>
            <p className="mt-5 max-w-xl text-base text-text-muted md:text-lg">
              Multi-asset dashboard, risk analytics, and execution workflows in one operator cockpit.
              Built for disciplined teams, not hype.
            </p>

            <div className="mt-8 grid gap-3 sm:grid-cols-2">
              <FeatureItem icon={Zap} title="Fast execution flows" description="Order lifecycle, status tracking, and emergency stop controls." />
              <FeatureItem icon={ShieldAlert} title="Risk-first defaults" description="Clear warnings, audit trail hooks, and policy-driven behavior." />
              <FeatureItem icon={BadgeCheck} title="Compliance-ready UX" description="Risk disclosure and legal routes integrated in product flow." />
              <FeatureItem icon={ArrowRight} title="Operational metrics" description="Portfolio, market, and PnL monitoring in real time." />
            </div>
          </div>

          <div className="rounded-2xl border border-border bg-surface/90 p-6 shadow-2xl shadow-black/30">
            <h2 className="font-['Sora'] text-2xl font-bold">Join Beta Waitlist</h2>
            <p className="mt-2 text-sm text-text-muted">
              Early access for operators and quant teams. No payment required.
            </p>

            <form onSubmit={handleSubmit} className="mt-6 space-y-4">
              <div>
                <label htmlFor="waitlist-email" className="mb-1 block text-sm text-text-muted">
                  Work email
                </label>
                <input
                  id="waitlist-email"
                  type="email"
                  required
                  autoComplete="email"
                  aria-describedby="waitlist-note"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  className="w-full rounded-lg border border-border bg-background px-3 py-2 text-text outline-none transition focus:border-primary"
                  placeholder="name@company.com"
                />
              </div>

              <div>
                <label htmlFor="waitlist-source" className="mb-1 block text-sm text-text-muted">
                  Source
                </label>
                <select
                  id="waitlist-source"
                  value={source}
                  onChange={(event) => setSource(event.target.value)}
                  className="w-full rounded-lg border border-border bg-background px-3 py-2 text-text outline-none transition focus:border-primary"
                >
                  <option value="landing_react">Landing page</option>
                  <option value="linkedin">LinkedIn</option>
                  <option value="twitter">X / Twitter</option>
                  <option value="referral">Referral</option>
                </select>
              </div>

              <button
                type="submit"
                disabled={joinWaitlist.isPending}
                className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2 font-semibold text-white transition hover:bg-primary/80 disabled:cursor-not-allowed disabled:opacity-70"
              >
                {joinWaitlist.isPending ? 'Submitting...' : 'Join waitlist'}
                {!joinWaitlist.isPending && <ArrowRight className="h-4 w-4" />}
              </button>
            </form>

            {signupMessage && (
              <div
                role="status"
                aria-live="polite"
                className={`mt-4 rounded-lg border px-3 py-2 text-sm ${
                  joinWaitlist.isError
                    ? 'border-danger/50 bg-danger/10 text-danger'
                    : 'border-success/50 bg-success/10 text-success'
                }`}
              >
                {signupMessage}
              </div>
            )}

            <div className="mt-5 text-sm text-text-muted">
              {typeof waitlistCount?.count === 'number'
                ? `${waitlistCount.count} users already joined the waitlist.`
                : 'Live waitlist counter unavailable.'}
            </div>
            <p className="mt-4 text-xs text-text-muted">
              <span id="waitlist-note">
              Trading involves risk and may result in total capital loss. This platform provides tools and analytics,
              not investment advice.
              </span>
            </p>
          </div>
        </section>
      </main>
      <footer className="border-t border-border/70 px-6 py-5">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-4 text-xs text-text-muted">
          <Link to="/legal/terms" className="hover:text-text">Terms</Link>
          <Link to="/legal/privacy" className="hover:text-text">Privacy</Link>
          <Link to="/legal/risk" className="hover:text-text">Risk Disclosure</Link>
          <a href="mailto:legal@ai-trading-system.com" className="hover:text-text">Contact</a>
          <span>Educational/demo mode may use simulated data.</span>
        </div>
      </footer>
    </div>
  );
}

function FeatureItem({
  icon: Icon,
  title,
  description,
}: {
  icon: React.ElementType;
  title: string;
  description: string;
}) {
  return (
    <div className="rounded-xl border border-border bg-surface/70 p-4">
      <div className="mb-2 inline-flex rounded-lg bg-primary/15 p-2 text-primary">
        <Icon className="h-4 w-4" />
      </div>
      <h3 className="font-['Sora'] text-sm font-semibold">{title}</h3>
      <p className="mt-1 text-sm text-text-muted">{description}</p>
    </div>
  );
}
