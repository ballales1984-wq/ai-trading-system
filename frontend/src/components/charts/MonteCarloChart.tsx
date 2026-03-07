import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';

interface MonteCarloChartProps {
  data: {
    percentile: string;
    value: number;
  }[];
  height?: number;
}

export default function MonteCarloChart({ data, height = 300 }: MonteCarloChartProps) {
  const option = useMemo(() => {
    // Extract values for calculating statistics
    const values = data.map(d => d.value);
    const p10 = values.sort((a, b) => a - b)[Math.floor(values.length * 0.1)];
    const p50 = values.sort((a, b) => a - b)[Math.floor(values.length * 0.5)];
    const p90 = values.sort((a, b) => a - b)[Math.floor(values.length * 0.9)];

    return {
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#161b22',
        borderColor: '#30363d',
        textStyle: {
          color: '#c9d1d9',
        },
        formatter: (params: { axisValue: string; value: number }[]) => {
          const item = params[0];
          return `
            <div style="padding: 4px;">
              <div style="font-weight: 600; margin-bottom: 4px;">${item.axisValue}</div>
              <div>Final Equity: <span style="color: #58a6ff; font-weight: 600;">$${item.value.toLocaleString()}</span></div>
            </div>
          `;
        },
      },
      grid: {
        left: '60px',
        right: '40px',
        top: '60px',
        bottom: '60px',
      },
      xAxis: {
        type: 'category',
        data: data.map(d => d.percentile),
        axisLine: {
          lineStyle: {
            color: '#30363d',
          },
        },
        axisLabel: {
          color: '#8b949e',
          fontSize: 10,
          rotate: 45,
        },
        axisTick: {
          show: false,
        },
      },
      yAxis: {
        type: 'value',
        axisLine: {
          show: false,
        },
        axisLabel: {
          color: '#8b949e',
          fontSize: 11,
          formatter: (value: number) => `$${(value / 1000).toFixed(0)}k`,
        },
        splitLine: {
          lineStyle: {
            color: '#21262d',
          },
        },
      },
      visualMap: {
        show: false,
        dimension: 0,
        pieces: [
          { lt: 2, color: '#da3633' },    // Bottom 10% - Very bad
          { gte: 2, lt: 4, color: '#f85149' },  // 10-25% - Bad
          { gte: 4, lt: 6, color: '#f0883e' },  // 25-40% - Below average
          { gte: 6, lt: 8, color: '#d29922' },  // 40-60% - Average
          { gte: 8, lt: 10, color: '#3fb950' }, // 60-75% - Good
          { gte: 10, color: '#238636' },        // Top 25% - Excellent
        ],
      },
      series: [
        {
          name: 'Monte Carlo',
          type: 'bar',
          data: data.map(d => d.value),
          barWidth: '80%',
          itemStyle: {
            borderRadius: [4, 4, 0, 0],
          },
          markLine: {
            silent: true,
            symbol: 'none',
            data: [
              {
                name: 'P10 (Worst)',
                xAxis: 1,
                lineStyle: {
                  color: '#da3633',
                  type: 'dashed',
                },
                label: {
                  show: true,
                  position: 'end',
                  color: '#da3633',
                  formatter: `P10: $${p10.toLocaleString()}`,
                },
              },
              {
                name: 'Median',
                xAxis: 5,
                lineStyle: {
                  color: '#58a6ff',
                  type: 'solid',
                },
                label: {
                  show: true,
                  position: 'end',
                  color: '#58a6ff',
                  formatter: `Median: $${p50.toLocaleString()}`,
                },
              },
              {
                name: 'P90 (Best)',
                xAxis: 9,
                lineStyle: {
                  color: '#3fb950',
                  type: 'dashed',
                },
                label: {
                  show: true,
                  position: 'end',
                  color: '#3fb950',
                  formatter: `P90: $${p90.toLocaleString()}`,
                },
              },
            ],
          },
          markPoint: {
            data: [
              {
                name: 'Median',
                value: p50,
                xAxis: 5,
                yAxis: p50,
                itemStyle: {
                  color: '#58a6ff',
                },
                label: {
                  show: true,
                  position: 'top',
                  color: '#c9d1d9',
                  formatter: `$${(p50 / 1000).toFixed(1)}k`,
                },
              },
            ],
          },
        },
      ],
    };
  }, [data]);

  const p50 = data[5]?.value || 0;
  const p10 = data[1]?.value || 0;
  const p90 = data[9]?.value || 0;

  return (
    <div>
      {/* Stats Summary */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="bg-background rounded-lg p-3 text-center">
          <div className="text-text-muted text-xs mb-1">P10 (Worst Case)</div>
          <div className="text-danger font-semibold">${p10.toLocaleString()}</div>
        </div>
        <div className="bg-background rounded-lg p-3 text-center">
          <div className="text-text-muted text-xs mb-1">Median</div>
          <div className="text-primary font-semibold">${p50.toLocaleString()}</div>
        </div>
        <div className="bg-background rounded-lg p-3 text-center">
          <div className="text-text-muted text-xs mb-1">P90 (Best Case)</div>
          <div className="text-success font-semibold">${p90.toLocaleString()}</div>
        </div>
      </div>
      
      <ReactECharts
        option={option}
        style={{ height: `${height}px`, width: '100%' }}
        opts={{ renderer: 'svg' }}
      />
    </div>
  );
}
