import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';

interface DrawdownChartProps {
  data: {
    date: string;
    drawdown: number;
    equity: number;
  }[];
  height?: number;
}

export default function DrawdownChart({ data, height = 300 }: DrawdownChartProps) {
  const option = useMemo(() => ({
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#161b22',
      borderColor: '#30363d',
      textStyle: {
        color: '#c9d1d9',
      },
      axisPointer: {
        type: 'line',
        lineStyle: {
          color: '#8b949e',
        },
      },
      formatter: (params: { axisValue: string; seriesName: string; value: number }[]) => {
        let html = `<div style="padding: 4px;"><div style="font-weight: 600; margin-bottom: 4px;">${params[0].axisValue}</div>`;
        
        params.forEach(param => {
          if (param.seriesName === 'Drawdown') {
            const color = param.value < -10 ? '#da3633' : param.value < -5 ? '#f85149' : '#f0883e';
            html += `<div>Drawdown: <span style="color: ${color}; font-weight: 600;">${param.value.toFixed(2)}%</span></div>`;
          } else if (param.seriesName === 'Equity' && param.value) {
            html += `<div>Equity: <span style="font-weight: 600;">$${param.value.toLocaleString()}</span></div>`;
          }
        });
        
        html += '</div>';
        return html;
      },
    },
    legend: {
      data: ['Drawdown', 'Equity'],
      top: 0,
      textStyle: {
        color: '#8b949e',
        fontSize: 11,
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
      },
      axisTick: {
        show: false,
      },
    },
    yAxis: [
      {
        type: 'value',
        name: 'Drawdown %',
        nameTextStyle: {
          color: '#8b949e',
          fontSize: 10,
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
      {
        type: 'value',
        name: 'Equity',
        nameTextStyle: {
          color: '#8b949e',
          fontSize: 10,
        },
        axisLine: {
          show: false,
        },
        axisLabel: {
          color: '#8b949e',
          fontSize: 11,
          formatter: (value: number) => `$${(value / 1000).toFixed(0)}k`,
        },
        splitLine: {
          show: false,
        },
      },
    ],
    series: [
      {
        name: 'Drawdown',
        type: 'line',
        data: data.map(d => d.drawdown),
        smooth: true,
        symbol: 'none',
        lineStyle: {
          color: '#f85149',
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
              { offset: 0, color: 'rgba(248, 81, 73, 0.4)' },
              { offset: 1, color: 'rgba(248, 81, 73, 0.05)' },
            ],
          },
        },
        markArea: {
          silent: true,
          itemStyle: {
            color: 'rgba(218, 54, 51, 0.1)',
          },
          data: [
            [
              {
                name: 'Severe Drawdown',
                yAxis: -20,
              },
              {
                yAxis: -100,
              },
            ],
          ],
        },
      },
      {
        name: 'Equity',
        type: 'line',
        yAxisIndex: 1,
        data: data.map(d => d.equity),
        smooth: true,
        symbol: 'none',
        lineStyle: {
          color: '#58a6ff',
          width: 1.5,
        },
      },
    ],
  }), [data]);

  return (
    <ReactECharts
      option={option}
      style={{ height: `${height}px`, width: '100%' }}
      opts={{ renderer: 'svg' }}
    />
  );
}
