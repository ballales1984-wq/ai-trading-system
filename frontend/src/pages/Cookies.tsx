import { Link } from 'react-router-dom';

export default function Cookies() {
  return (
    <div className="min-h-screen bg-bg-primary text-text py-20 px-4">
      <div className="max-w-3xl mx-auto">
        <Link to="/" className="text-primary hover:underline mb-8 inline-block">← Back to Home</Link>
        <h1 className="text-4xl font-bold mb-8">Cookie Policy</h1>
        <p className="text-text-muted mb-8">Last updated: April 8, 2026</p>

        <div className="prose prose-invert max-w-none space-y-6">
          <h2 className="text-2xl font-semibold text-white">1. What Are Cookies</h2>
          <p className="text-text-muted">
            Cookies are small text files stored on your device when you visit websites. They help remember your preferences and improve your experience.
          </p>

          <h2 className="text-2xl font-semibold text-white">2. How We Use Cookies</h2>
          <p className="text-text-muted">
            We use cookies for:
          </p>
          <ul className="list-disc pl-6 text-text-muted space-y-2">
            <li>Essential functionality and security</li>
            <li>Analytics to improve our service</li>
            <li>Remembering your preferences</li>
          </ul>

          <h2 className="text-2xl font-semibold text-white">3. Managing Cookies</h2>
          <p className="text-text-muted">
            You can control or delete cookies through your browser settings. Please note that disabling cookies may affect the functionality of our service.
          </p>

          <h2 className="text-2xl font-semibold text-white">4. Third-Party Cookies</h2>
          <p className="text-text-muted">
            We may use third-party services (including Google Analytics and Google AdSense) that set their own cookies. These are subject to their respective privacy policies.
          </p>

          <h2 className="text-2xl font-semibold text-white">5. Contact Us</h2>
          <p className="text-text-muted">
            For questions about this Cookie Policy, please contact us at support@ai-trading-system.com
          </p>
        </div>
      </div>
    </div>
  );
}
