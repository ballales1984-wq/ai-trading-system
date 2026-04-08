import { Link } from 'react-router-dom';

export default function Privacy() {
  return (
    <div className="min-h-screen bg-bg-primary text-text py-20 px-4">
      <div className="max-w-3xl mx-auto">
        <Link to="/" className="text-primary hover:underline mb-8 inline-block">← Back to Home</Link>
        <h1 className="text-4xl font-bold mb-8">Privacy Policy</h1>
        <p className="text-text-muted mb-8">Last updated: April 8, 2026</p>

        <div className="prose prose-invert max-w-none space-y-6">
          <h2 className="text-2xl font-semibold text-white">1. Introduction</h2>
          <p className="text-text-muted">
            AI Trading System ("we", "our", or "us") operates the ai-trading-system-1reg.onrender.com website. 
            This Privacy Policy describes how we collect, use, and share information when you use our service.
          </p>

          <h2 className="text-2xl font-semibold text-white">2. Information We Collect</h2>
          <p className="text-text-muted">
            We collect information you provide directly to us, including:
          </p>
          <ul className="list-disc pl-6 text-text-muted space-y-2">
            <li>Email address for waitlist and account registration</li>
            <li>Trading preferences and risk tolerance data</li>
            <li>Usage data and analytics</li>
          </ul>

          <h2 className="text-2xl font-semibold text-white">3. How We Use Your Information</h2>
          <p className="text-text-muted">
            We use the information we collect to:
          </p>
          <ul className="list-disc pl-6 text-text-muted space-y-2">
            <li>Provide and maintain our trading services</li>
            <li>Send you updates and marketing communications</li>
            <li>Improve our services and user experience</li>
            <li>Comply with legal obligations</li>
          </ul>

          <h2 className="text-2xl font-semibold text-white">4. Data Security</h2>
          <p className="text-text-muted">
            We implement appropriate technical and organizational measures to protect your personal data against unauthorized access, alteration, disclosure, or destruction.
          </p>

          <h2 className="text-2xl font-semibold text-white">5. Contact Us</h2>
          <p className="text-text-muted">
            If you have questions about this Privacy Policy, please contact us at support@ai-trading-system.com
          </p>

          <h2 className="text-2xl font-semibold text-white">6. Cookies</h2>
          <p className="text-text-muted">
            We also use cookies. For more information about how we use cookies, please see our <Link to="/cookies" className="text-primary hover:underline">Cookie Policy</Link>.
          </p>
        </div>
      </div>
    </div>
  );
}
