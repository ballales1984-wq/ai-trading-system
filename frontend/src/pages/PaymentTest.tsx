import { useState } from 'react';
import { paymentApi } from '../services/api';

export default function PaymentTest() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [priceId, setPriceId] = useState<string>('');

  const handlePaymentLinkTest = () => {
    if (paymentApi.isConfigured()) {
      paymentApi.redirectToPaymentLink();
    } else {
      setError('Stripe Payment Link non configurato. Imposta VITE_STRIPE_PAYMENT_LINK');
    }
  };

  const handleCheckoutTest = async () => {
    // Basic validation
    if (!priceId) {
      setError('Please enter a Price ID');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await paymentApi.createCheckoutSession({
        price_id: priceId || undefined, // Use provided or default from env
        quantity: 1,
      });
      
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

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">🧪 Test Pagamento Stripe</h1>
      
      <div className="space-y-4">
        {/* Status */}
        <div className={`p-4 rounded-lg ${isPaymentLinkConfigured ? 'bg-green-900/30' : 'bg-yellow-900/30'}`}>
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
              <label htmlFor="price-id" className="block text-sm font-medium text-slate-300 mb-2">
                Price ID (optional - uses default from env if empty)
              </label>
              <input
                id="price-id"
                type="text"
                value={priceId}
                onChange={(e) => setPriceId(e.target.value)}
                className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="price_123abc..."
              />
            </div>
            <button
              onClick={handleCheckoutTest}
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-4 py-2 rounded-lg font-medium"
            >
              {loading ? 'Creazione sessione...' : 'Test Checkout Session'}
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
          >
            Test Payment Link
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-900/30 border border-red-700 rounded-lg p-4">
            <p className="text-red-400">❌ {error}</p>
          </div>
        )}

        {/* Success Display */}
        {success && (
          <div className="bg-green-900/30 border border-green-700 rounded-lg p-4">
            <p className="text-green-400">✅ {success}</p>
          </div>
        )}
      </div>
    </div>
  );
}

