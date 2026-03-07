import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';

interface RiskReturnScatterProps {
  data: {
    name: string;
    risk: number;      // Standard deviation / volatility
    return: number;    // Expected return
    color?: string;
  }[];
  height?: number;
}

export default function RiskReturnScatter({ data, height = 350 }: RiskReturnScatterProps) {
  const option = useMemo(() => ({
    tooltip: {
      trigger: 'item',
      backgroundColor: '#161b22',
      borderColor: '#30363d',
      textStyle: {
        color: '#c9d1d9',
      },
      formatter: (params: { data: [string, number, number] }) => {
        const [name, risk, ret] = params.data;
        return `
          <div style="padding: 4px;">
            <div style="font-weight: 600; margin-bottom: 4px;">${name}</div>
            <div>Risk (σ): <span style="color: #f85149; font-weight: 600;">${risk.toFixed(2)}%</span></div>
            <div>Return: <span style="color: #3fb950; font-weight: 600;">${ret.toFixed(2)}%</span></div>
            <div>Sharpe: <span style="color: #58a6ff; font-weight: 600;">${(ret / risk).toFixed(2)}</span></div>
          </div>
        `;
      },
    },
    grid: {
      left: '60px',
      right: '40px',
      top: '40px',
      bottom: '60px',
    },
    xAxis: {
      type: 'value',
      name: 'Risk (Volatility %)',
      nameLocation: 'middle',
      nameGap: 35,
      nameTextStyle: {
        color: '#8b949e',
        fontSize: 11,
      },
      axisLine: {
        lineStyle: {
          color: '#30363d',
        },
      },
      axisLabel: {
        color: '#8b949e',
        fontSize: 11,
        formatter: '{value}%',
      },
      splitLine: {
        lineStyle: {
          color: '#21262d',
        },
      },
    },
    yAxis: {
      type: 'value',
      name: 'Return (%)',
      nameLocation: 'middle',
      nameGap: 45,
      nameTextStyle: {
        color: '#8b949e',
        fontSize: 11,
      },
      axisLine: {
        show: false,
      },
      axisLabel: {
        color: '#8b949e',
        fontSize: 11,
        formatter: '{value}%',
      },
      splitLine: {
        lineStyle: {
          color: '#21262d',
        },
      },
    },
    series: [
      {
        name: 'Assets',
        type: 'scatter',
        data: data.map(d => [d.risk, d.return, d.name]),
        symbolSize: (val: number[]) => {
          // Scale symbol size based on return
          const size = Math.max(12, Math.min(30, Math.abs(val[1]) * 2 + 12));
          return size;
        },
        itemStyle: {
          color: (params: { data: [string, number, number] }) => {
            const [, risk, ret] = params.data;
            // Color based on risk-adjusted return
            const sharpe = ret / risk;
            if (sharpe > 1.5) return '#3fb950';
            if (sharpe > 1) return '#58a6ff';
            if (sharpe > 0.5) return '#d29922';
            if (sharpe > 0) return '#f0883e';
            return '#f85149';
          },
          opacity: 0.8,
          borderColor: '#161b22',
          borderWidth: 2,
        },
        emphasis: {
          itemStyle: {
            opacity: 1,
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.5)',
          },
        },
        label: {
          show: true,
          position: 'top',
          formatter: (params: { data: [string, number, number] }) => params.data[0],
          color: '#c9d1d9',
          fontSize: 10,
          fontWeight: 500,
        },
        markLine: {
          silent: true,
          symbol: 'none',
          lineStyle: {
            color: '#8b949e',
            type: 'dashed',
          },
          data: [
            {
              name: 'Capital Market Line',
              type: 'average',
              xAxis: 'mean',
            },
          ],
          label: {
            show: true,
            position: 'end',
            color: '#8b949e',
            formatter: 'Avg Risk',
          },
        },
      },
      // Efficient Frontier curve (simplified)
      {
        name: 'Efficient Frontier',
        type: 'line',
        smooth: true,
        symbol: 'none',
        lineStyle: {
          color: '#8b949e',
          type: 'dashed',
          width: 1,
          opacity: 0.5,
        },
        data: [
          [5, 8], [10, 12], [15, 15], [20, 17], [25, 18.5], [30, 19.5], [35, 20]
        ].map(([r, ret]) => [r, ret]),
      },
    ],
    visualMap: {
      show: false,
    },
  }), [data]);

  // Calculate portfolio metrics
  const avgRisk = data.reduce((sum, d) => sum + d.risk, 0) / data.length;
  const avgReturn = data.reduce((sum, d) => sum + d.return, 0) / data.length;
  const bestSharpe = Math.max(...data.map(d => d.return / d.risk));

  return (
    <div>
      {/* Stats Summary */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="bg-background rounded-lg p-3 text-center">
          <div className="text-text-muted text-xs mb-1">Avg Risk</div>
          <div className="text-danger font-semibold">{avgRisk.toFixed(2)}%</div>
        </div>
        <div className="bg-background rounded-lg p-3 text-center">
          <div className="text-text-muted text-xs mb-1">Avg Return</div>
          <div className="text-success font-semibold">{avgReturn.toFixed(2)}%</div>
        </div>
        <div className="bg-background rounded-lg p-3 text-center">
          <div className="text-text-muted text-xs mb-1">Best Sharpe</div>
          <div className="text-primary font-semibold">{bestSharpe.toFixed(2)}</div>
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
