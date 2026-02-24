export default function Terms() {
  return (
    <div className="p-6 max-w-4xl">
      <h1 className="text-2xl font-bold text-text mb-3">Terms of Service</h1>
      <p className="text-text-muted mb-6">
        Last updated: February 23, 2026
      </p>

      <div className="space-y-5 text-sm leading-6 text-text-muted">
        <section>
          <h2 className="text-text font-semibold mb-2">Service scope</h2>
          <p>
            This platform provides analytics, portfolio dashboards, and order management tools. It does not provide
            personalized financial, investment, legal, or tax advice.
          </p>
        </section>

        <section>
          <h2 className="text-text font-semibold mb-2">Trading risk</h2>
          <p>
            Trading digital assets, derivatives, and leveraged products involves substantial risk. You can lose part
            or all of your capital. Past performance does not guarantee future results.
          </p>
        </section>

        <section>
          <h2 className="text-text font-semibold mb-2">Execution and availability</h2>
          <p>
            The service may be unavailable due to maintenance, provider outages, API limits, or network interruptions.
            Order execution may fail, be delayed, or differ from requested parameters.
          </p>
        </section>

        <section>
          <h2 className="text-text font-semibold mb-2">User responsibilities</h2>
          <p>
            You are responsible for account security, API keys, configuration choices, and compliance with laws
            applicable to your jurisdiction.
          </p>
        </section>
      </div>
    </div>
  );
}
