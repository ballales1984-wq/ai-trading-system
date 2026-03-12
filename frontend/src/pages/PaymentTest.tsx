import { useState } from 'react';
import { paymentApi } from '../services/api';

export default function PaymentTest() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [email, setEmail] = useState<string>('test@example.com');

  const handlePaymentLinkTest = () => {
    if (paymentApi.isConfigured()) {
      paymentApi.redirectToPaymentLink();
    } else {
      setError('Stripe Payment Link non configurato. Imposta VITE_STRIPE_PAYMENT_LINK');
    }
  };

  const handleCheckoutTest = async () => {
    // Basic validation
    if (!email) {
      setError('Please enter an email address');
      return;
    }

    // Simple email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setError('Please enter a valid email address');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await paymentApi.createCheckoutSession(email);
      
      setSuccess(`Session created! Redirecting to: ${response.checkout_url}`);
      
      // Redirect to Stripe
      window.location.href = response.checkout_url;
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Errore nel pagamento');
    } finally {
      setLoading(false);
    }
  };

  const isPaymentLinkConfigured = paymentApi.isConfigured();

  // Handle Enter key press for form submission
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !loading) {
      e.preventDefault();
      handleCheckoutTest();
    }
  };

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-6" id="payment-test-heading">
        🧪 Test Pagamento Stripe
      </h1>
      
      <div className="space-y-4">
        {/* Status */}
        <div 
          className={`p-4 rounded-lg ${isPaymentLinkConfigured ? 'bg-green-900/30' : 'bg-yellow-900/30'}`}
          role="status"
          aria-live="polite"
        >
          <p className="font-medium">
            Status: {isPaymentLinkConfigured ? '✅ Payment Link configurato' : '⚠️ Payment Link non configurato'}
          </p>
          <p className="text-sm text-gray-400 mt-1">
            Usa il metodo 2 se hai un Payment Link, altrimenti usa il metodo 1 (Checkout Session)
          </p>
        </div>

        {/* Method 1: Checkout Session */}
        <div className="border border-gray-700 rounded-lg p-4">
          <h2 className="text-lg font-semibold mb-2">Metodo 1: Stripe Checkout Session</h2>
          <p className="text-sm text-gray-400 mb-4">
            Crea una sessione di pagamento tramite backend API
          </p>
          <div className="space-y-3">
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-slate-300 mb-2">
                Email for checkout session
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                onKeyDown={handleKeyDown}
                className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="test@example.com"
                aria-required="true"
                aria-describedby="email-help"
              />
              <p id="email-help" className="text-xs text-text-muted mt-1">
                Used to create a Stripe checkout session
              </p>
            </div>
            <button
              onClick={handleCheckoutTest}
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-4 py-2 rounded-lg font-medium"
              aria-disabled={String(loading)}
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Creazione sessione...
                </span>
              ) : (
                'Test Checkout Session'
              )}
            </button>
          </div>
        </div>

        {/* Method 2: Payment Link */}
        <div className="border border-gray-700 rounded-lg p-4">
          <h2 className="text-lg font-semibold mb-2">Metodo 2: Stripe Payment Link</h2>
          <p className="text-sm text-gray-400 mb-4">
            Redirect diretto a Stripe Payment Link (più semplice)
          </p>
          <button
            onClick={handlePaymentLinkTest}
            disabled={!isPaymentLinkConfigured}
            className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 px-4 py-2 rounded-lg font-medium"
            aria-disabled={String(!isPaymentLinkConfigured)}
            aria-label={isPaymentLinkConfigured ? 'Test Payment Link' : 'Payment Link not configured'}
          >
            Test Payment Link
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div 
            className="bg-red-900/30 border border-red-700 rounded-lg p-4"
            role="alert"
          >
            <p className="text-red-400">❌ {error}</p>
          </div>
        )}

        {/* Success Display */}
        {success && (
          <div 
            className="bg-green-900/30 border border-green-700 rounded-lg p-4"
            role="status"
            aria-live="polite"
          >
            <p className="text-green-400">✅ {success}</p>
          </div>
        )}
      </div>
    </div>
  );
}

