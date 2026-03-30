import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';

interface RollingSharpeChartProps {
  data: {
    date: string;
    rolling_sharpe: number;
  }[];
  height?: number;
}

export default function RollingSharpeChart({ data, height = 300 }: RollingSharpeChartProps) {
  // Guard against empty or invalid data
  if (!data || !Array.isArray(data) || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-text-muted">
        No rolling Sharpe data available
      </div>
    );
  }

  // Ensure minimum height is valid
  const chartHeight = Math.max(height || 300, 200);

  const option = useMemo(() => ({
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#161b22',
      borderColor: '#30363d',
      textStyle: {
        color: '#c9d1d9',
      },
      axisPointer: {
        type: 'cross',
        crossStyle: {
          color: '#8b949e',
        },
      },
      formatter: (params: { axisValue: string; value: number }[]) => {
        const item = params[0];
        const color = item.value >= 0 ? '#3fb950' : '#f85149';
        return `
          <div style="padding: 4px;">
            <div style="font-weight: 600; margin-bottom: 4px;">${item.axisValue}</div>
            <div>Rolling Sharpe: <span style="color: ${color}; font-weight: 600;">${item.value.toFixed(2)}</span></div>
          </div>
        `;
      },
    },
    grid: {
      left: '50px',
      right: '20px',
      top: '40px',
      bottom: '40px',
    },
    xAxis: {
      type: 'category',
      data: data.map(d => d.date),
      axisLine: {
        lineStyle: {
          color: '#30363d',
        },
      },
      axisLabel: {
        color: '#8b949e',
        fontSize: 10,
        formatter: (value: string) => {
          // Show only every nth label to avoid overcrowding
          return value;
        },
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
      },
      splitLine: {
        lineStyle: {
          color: '#21262d',
        },
      },
    },
    visualMap: {
      show: false,
      pieces: [
        { gt: 0, color: '#3fb950' },
        { lte: 0, color: '#f85149' },
      ],
    },
    series: [
      {
        name: 'Rolling Sharpe',
        type: 'line',
        data: data.map(d => d.rolling_sharpe),
        smooth: true,
        symbol: 'none',
        lineStyle: {
          width: 2,
        },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(63, 185, 80, 0.3)' },
              { offset: 1, color: 'rgba(63, 185, 80, 0.05)' },
            ],
          },
        },
        markLine: {
          silent: true,
          lineStyle: {
            color: '#8b949e',
            type: 'dashed',
          },
          data: [
            { yAxis: 0, label: { show: false } },
            { yAxis: 1, label: { show: false } },
            { yAxis: 2, label: { show: false } },
          ],
        },
      },
    ],
  }), [data]);

  return (
    <div style={{ minHeight: `${chartHeight}px`, minWidth: '100%' }}>
      <ReactECharts
        option={option}
        style={{ height: `${chartHeight}px`, width: '100%' }}
        opts={{ renderer: 'svg' }}
      />
    </div>
  );
}
