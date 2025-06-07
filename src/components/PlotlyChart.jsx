import React, { useEffect, useState } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

const PlotlyChart = ({ data, layout }) => {
  const [Plot, setPlot] = useState(null);

  useEffect(() => {
    let mounted = true;
    import('react-plotly.js').then((module) => {
      if (mounted) {
        setPlot(() => module.default);
      }
    });
    return () => {
      mounted = false;
    };
  }, []);

  if (!Plot) return <div>Loading chart...</div>;

  return (
    <Plot
      data={data}
      layout={layout}
      useResizeHandler
      style={{ width: '100%', height: '100%' }}
    />
  );
};

const PlotlyChartWrapper = (props) => (
  <BrowserOnly fallback={<div>Loading...</div>}>
    {() => <PlotlyChart {...props} />}
  </BrowserOnly>
);

export default PlotlyChartWrapper;
