export default function RiskDisclosure() {
  return (
    <div className="p-6 max-w-4xl">
      <h1 className="text-2xl font-bold text-text mb-3">Risk Disclosure</h1>
      <p className="text-text-muted mb-6">
        Last updated: February 23, 2026
      </p>

      <div className="space-y-5 text-sm leading-6 text-text-muted">
        <section>
          <h2 className="text-text font-semibold mb-2">High risk warning</h2>
          <p>
            Trading in crypto-assets and other volatile instruments is high risk. Prices can move rapidly and
            unpredictably, causing significant losses.
          </p>
        </section>

        <section>
          <h2 className="text-text font-semibold mb-2">No guaranteed returns</h2>
          <p>
            Strategies, AI signals, and backtests are informational and probabilistic. They do not guarantee profits
            or capital preservation.
          </p>
        </section>

        <section>
          <h2 className="text-text font-semibold mb-2">Operational risks</h2>
          <p>
            Technical failures may include connectivity loss, stale data, delayed execution, rejected orders, and
            external exchange downtime.
          </p>
        </section>

        <section>
          <h2 className="text-text font-semibold mb-2">Suitability</h2>
          <p>
            Before using live trading, evaluate your objectives and risk tolerance, and seek independent professional
            advice where required.
          </p>
        </section>
      </div>
    </div>
  );
}
