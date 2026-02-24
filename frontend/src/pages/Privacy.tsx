export default function Privacy() {
  return (
    <div className="p-6 max-w-4xl">
      <h1 className="text-2xl font-bold text-text mb-3">Privacy Policy</h1>
      <p className="text-text-muted mb-6">
        Last updated: February 23, 2026
      </p>

      <div className="space-y-5 text-sm leading-6 text-text-muted">
        <section>
          <h2 className="text-text font-semibold mb-2">Data we process</h2>
          <p>
            We process account identifiers, portfolio data, order events, and technical telemetry required for
            operation, monitoring, and security.
          </p>
        </section>

        <section>
          <h2 className="text-text font-semibold mb-2">Purpose</h2>
          <p>
            Data is used to provide dashboard features, risk controls, troubleshooting, and fraud/security prevention.
          </p>
        </section>

        <section>
          <h2 className="text-text font-semibold mb-2">Retention</h2>
          <p>
            Data is retained only as needed for product functionality, legal obligations, and incident investigation.
          </p>
        </section>

        <section>
          <h2 className="text-text font-semibold mb-2">Your rights</h2>
          <p>
            Depending on jurisdiction, you may request access, correction, export, or deletion of your personal data.
          </p>
        </section>
      </div>
    </div>
  );
}
