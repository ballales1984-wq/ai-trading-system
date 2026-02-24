import { FormEvent, useMemo, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useMutation } from '@tanstack/react-query';
import { paymentsApi } from '../services/api';

const DEFAULT_ACCESS_CODE = 'BETA-ACCESS-2026';

function useNextPath(): string {
  const location = useLocation();
  const params = new URLSearchParams(location.search);
  const next = params.get('next');
  if (!next || !next.startsWith('/')) return '/dashboard';
  return next;
}

function usePaidFlag(): boolean {
  const location = useLocation();
  const params = new URLSearchParams(location.search);
  return params.get('paid') === '1';
}

export default function AccessGate() {
  const navigate = useNavigate();
  const nextPath = useNextPath();
  const paid = usePaidFlag();
  const expectedCode = (import.meta.env.VITE_DASHBOARD_ACCESS_CODE || DEFAULT_ACCESS_CODE).trim();
  const [email, setEmail] = useState('');
  const [code, setCode] = useState('');
  const [appDownloaded, setAppDownloaded] = useState(false);
  const [error, setError] = useState('');
  const [checkoutError, setCheckoutError] = useState('');

  const stripeCheckout = useMutation({
    mutationFn: (customerEmail: string) =>
      paymentsApi.createStripeCheckoutSession({ email: customerEmail }),
    onSuccess: (data) => {
      window.location.href = data.checkout_url;
    },
    onError: (err: unknown) => {
      if (
        typeof err === 'object' &&
        err &&
        'response' in err &&
        typeof (err as { response?: { data?: { detail?: string } } }).response?.data?.detail === 'string'
      ) {
        setCheckoutError((err as { response: { data: { detail: string } } }).response.data.detail);
      } else {
        setCheckoutError('Unable to open Stripe checkout right now.');
      }
    },
  });

  const canSubmit = useMemo(
    () => email.trim().length > 3 && code.trim().length > 0 && appDownloaded,
    [appDownloaded, code, email],
  );

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!canSubmit) return;

    if (code.trim() !== expectedCode) {
      setError('Invalid access code.');
      return;
    }

    window.localStorage.setItem('ats_access_granted', '1');
    window.localStorage.setItem('ats_app_download_confirmed', '1');
    window.localStorage.setItem('ats_access_email', email.trim());
    navigate(nextPath, { replace: true });
  };

  const openCheckout = () => {
    setCheckoutError('');
    if (!email.trim()) {
      setCheckoutError('Enter your email first.');
      return;
    }
    stripeCheckout.mutate(email.trim());
  };

  return (
    <div className="min-h-screen bg-background text-text flex items-center justify-center px-4">
      <div className="w-full max-w-md rounded-xl border border-border bg-surface p-6">
        <h1 className="text-2xl font-bold">Access Required</h1>
        <p className="mt-2 text-sm text-text-muted">
          Web dashboard access is limited. Login and confirm app download before continuing.
        </p>
        <p className="mt-2 text-xs text-text-muted">
          Payment is handled via Stripe. Price can be configured later in backend settings.
        </p>

        <form onSubmit={onSubmit} className="mt-5 space-y-4">
          <div>
            <label className="mb-1 block text-sm text-text-muted" htmlFor="access-email">
              Email
            </label>
            <input
              id="access-email"
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded border border-border bg-background px-3 py-2 text-text outline-none focus:border-primary"
            />
          </div>

          <div>
            <label className="mb-1 block text-sm text-text-muted" htmlFor="access-code">
              Access code
            </label>
            <input
              id="access-code"
              type="password"
              required
              value={code}
              onChange={(e) => setCode(e.target.value)}
              className="w-full rounded border border-border bg-background px-3 py-2 text-text outline-none focus:border-primary"
            />
          </div>

          <label className="flex items-start gap-2 text-sm text-text-muted">
            <input
              type="checkbox"
              checked={appDownloaded}
              onChange={(e) => setAppDownloaded(e.target.checked)}
              className="mt-1"
            />
            I confirm I downloaded the desktop app and accept restricted access policy.
          </label>

          <button
            type="button"
            onClick={openCheckout}
            disabled={stripeCheckout.isPending}
            className="w-full rounded border border-primary px-4 py-2 font-semibold text-primary disabled:opacity-60"
          >
            {stripeCheckout.isPending ? 'Opening Stripe...' : 'Pay with Stripe'}
          </button>

          {checkoutError && <p className="text-sm text-danger">{checkoutError}</p>}
          {paid && (
            <p className="text-sm text-success">
              Payment confirmed. Complete login + app download confirmation to continue.
            </p>
          )}

          {error && <p className="text-sm text-danger">{error}</p>}

          <button
            type="submit"
            disabled={!canSubmit}
            className="w-full rounded bg-primary px-4 py-2 font-semibold text-white disabled:opacity-60"
          >
            Continue to Dashboard
          </button>
        </form>

        <div className="mt-4 text-xs text-text-muted">
          Need access? Join waitlist on the <Link to="/" className="text-primary hover:underline">marketing page</Link>.
        </div>
      </div>
    </div>
  );
}
