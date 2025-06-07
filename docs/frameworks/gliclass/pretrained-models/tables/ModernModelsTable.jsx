import React, { useState } from 'react';

export default function BiEncoderTable() {
  const [data, setData] = useState([
    {
      name: '[gliclass-modern-large-v2.0](https://huggingface.co/knowledgator/gliclass-modern-large-v2.0)',
      encoder: '[ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large)',
      size: 1.6,
      f1: 0.6045,
    },
    {
      name: '[gliclass-modern-base-v2.0](https://huggingface.co/knowledgator/gliclass-modern-base-v2.0)',
      encoder: '[ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)',
      size: 0.694,
      f1: 0.5563,
    },
    {
      name: '[gliclass-modern-base-v2.0-init](https://huggingface.co/knowledgator/gliclass-modern-base-v2.0-init)',
      encoder: '[ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)',
      size: 0.606,
      f1: 0.5129,
    },
    {
      name: '[gliclass-modern-large-v2.0-init](https://huggingface.co/knowledgator/gliclass-modern-large-v2.0-init)',
      encoder: '[ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large)',
      size: 1.6,
      f1: 0.5447,
    },
  ]);

  const [sortKey, setSortKey] = useState('f1');
  const [ascending, setAscending] = useState(false);

  const sorted = [...data].sort((a, b) => {
    if (a[sortKey] < b[sortKey]) return ascending ? -1 : 1;
    if (a[sortKey] > b[sortKey]) return ascending ? 1 : -1;
    return 0;
  });

  const handleSort = (key) => {
    if (key === sortKey) setAscending(!ascending);
    else {
      setSortKey(key);
      setAscending(true);
    }
  };

  const renderSortHeader = (label, key) => {
    const isActive = sortKey === key;
    const arrow = ascending ? '▲' : '▼';
    return (
      <th
        onClick={() => handleSort(key)}
        style={{ cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap' }}
      >
        {label}
        <span style={{ visibility: isActive ? 'visible' : 'hidden', marginLeft: '4px' }}>
          {arrow}
        </span>
      </th>
    );
  };

  const renderMarkdownLink = (text) => {
    const match = text.match(/\[(.*?)\]\((.*?)\)/);
    if (!match) return text;
    return <a href={match[2]}>{match[1]}</a>;
  };

  return (
    <table>
      <thead>
        <tr>
          {renderSortHeader('Name', 'name')}
          {renderSortHeader('Encoder', 'encoder')}
          {renderSortHeader('Size (GB)', 'size')}
          {renderSortHeader('Zero-Shot F1 Score', 'f1')}
        </tr>
      </thead>
      <tbody>
        {sorted.map((row, index) => (
          <tr key={index}>
            <td>{renderMarkdownLink(row.name)}</td>
            <td>{renderMarkdownLink(row.encoder)}</td>
            <td>{row.size}</td>
            <td>{row.f1}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
