import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import type { CorrelationMatrix as CorrelationMatrixType } from '../../types';

interface CorrelationMatrixProps {
  data: CorrelationMatrixType;
  height?: number;
}

export default function CorrelationMatrix({ data, height = 350 }: CorrelationMatrixProps) {
  const { assets, matrix } = data;

  // Transform matrix data for ECharts heatmap
  const heatmapData = useMemo(() => {
    const result: [number, number, number][] = [];
    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < matrix[i].length; j++) {
        // ECharts heatmap: [x, y, value]
        // We swap i and j to match the visual layout (y on vertical axis)
        result.push([j, i, matrix[i][j]]);
      }
    }
    return result;
  }, [matrix]);

  const option = {
    tooltip: {
      position: 'top',
      backgroundColor: '#161b22',
      borderColor: '#30363d',
      textStyle: {
        color: '#c9d1d9',
      },
      formatter: (params: { data: [number, number, number] }) => {
        const [x, y, value] = params.data;
        const assetX = assets[x];
        const assetY = assets[y];
        let correlationColor = '#8b949e';
        if (value > 0.7) correlationColor = '#3fb950';
        else if (value > 0.3) correlationColor = '#58a6ff';
        else if (value < -0.3) correlationColor = '#f85149';
        else if (value < -0.7) correlationColor = '#da3633';
        
        return `
          <div style="padding: 4px;">
            <div style="font-weight: 600; margin-bottom: 4px;">${assetY} ↔ ${assetX}</div>
            <div>Correlation: <span style="color: ${correlationColor}; font-weight: 600;">${value.toFixed(2)}</span></div>
          </div>
        `;
      },
    },
    grid: {
      left: '80px',
      right: '40px',
      top: '40px',
      bottom: '80px',
    },
    xAxis: {
      type: 'category',
      data: assets,
      position: 'top',
      splitArea: {
        show: true,
      },
      axisLabel: {
        color: '#8b949e',
        fontSize: 11,
        rotate: 45,
      },
      axisLine: {
        show: false,
      },
      axisTick: {
        show: false,
      },
    },
    yAxis: {
      type: 'category',
      data: assets,
      splitArea: {
        show: true,
      },
      axisLabel: {
        color: '#8b949e',
        fontSize: 11,
      },
      axisLine: {
        show: false,
      },
      axisTick: {
        show: false,
      },
    },
    visualMap: {
      min: -1,
      max: 1,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: '0px',
      inRange: {
        // Red (negative) -> White (zero) -> Green (positive)
        color: ['#da3633', '#f85149', '#ffffff', '#3fb950', '#238636'],
      },
      textStyle: {
        color: '#8b949e',
      },
    },
    series: [
      {
        name: 'Correlation Matrix',
        type: 'heatmap',
        data: heatmapData,
        label: {
          show: true,
          color: '#c9d1d9',
          fontSize: 10,
          fontWeight: 500,
          formatter: (params: { value: [number, number, number] }) => {
            return params.value[2].toFixed(2);
          },
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.5)',
          },
        },
        itemStyle: {
          borderColor: '#161b22',
          borderWidth: 2,
        },
      },
    ],
  };

  return (
    <ReactECharts
      option={option}
      style={{ height: `${height}px`, width: '100%' }}
      opts={{ renderer: 'svg' }}
    />
  );
}

