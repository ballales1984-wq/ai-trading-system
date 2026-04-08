import { Link } from 'react-router-dom';

export default function Terms() {
  return (
    <div className="min-h-screen bg-bg-primary text-text py-20 px-4">
      <div className="max-w-3xl mx-auto">
        <Link to="/" className="text-primary hover:underline mb-8 inline-block">← Back to Home</Link>
        <h1 className="text-4xl font-bold mb-8">Terms of Service</h1>
        <p className="text-text-muted mb-8">Last updated: April 8, 2026</p>

        <div className="prose prose-invert max-w-none space-y-6">
          <h2 className="text-2xl font-semibold text-white">1. Acceptance of Terms</h2>
          <p className="text-text-muted">
            By accessing and using AI Trading System, you accept and agree to be bound by the terms and provision of this agreement.
          </p>

          <h2 className="text-2xl font-semibold text-white">2. Use License</h2>
          <p className="text-text-muted">
            Permission is granted to use our services for personal, non-commercial use only. This is the grant of a license, not a transfer of title.
          </p>

          <h2 className="text-2xl font-semibold text-white">3. Risk Disclaimer</h2>
          <p className="text-text-muted">
            Trading in financial markets involves substantial risk. Past performance does not guarantee future results. 
            The simulations and predictions provided by our system are for informational purposes only and should not be considered financial advice.
          </p>

          <h2 className="text-2xl font-semibold text-white">4. No Financial Advice</h2>
          <p className="text-text-muted">
            AI Trading System provides analytical tools and simulations only. We do not provide investment advice. 
            You are solely responsible for any investment decisions you make.
          </p>

          <h2 className="text-2xl font-semibold text-white">5. Limitation of Liability</h2>
          <p className="text-text-muted">
            In no event shall AI Trading System be liable for any damages arising out of the use or inability to use our services, 
            including but not limited to trading losses.
          </p>

          <h2 className="text-2xl font-semibold text-white">6. Contact Information</h2>
          <p className="text-text-muted">
            For questions about these Terms of Service, please contact us at support@ai-trading-system.com
          </p>
        </div>
      </div>
    </div>
  );
}
