import React, { useState } from 'react';

export default function BiEncoderTable() {
  const [data, setData] = useState([
    {
      name: '[gliclass-base-v1.0-lw](https://huggingface.co/knowledgator/gliclass-base-v1.0-lw)',
      encoder: '[deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base)',
      size: 0.749,
      f1: 0.6183,
    },
    {
      name: '[gliclass-large-v1.0-lw](https://huggingface.co/knowledgator/gliclass-large-v1.0-lw)',
      encoder: '[deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large)',
      size: 1.76,
      f1: 0.6165,
    },
    {
      name: '[gliclass-small-v1.0-lw](https://huggingface.co/knowledgator/gliclass-small-v1.0-lw)',
      encoder: '[deberta-v3-small](https://huggingface.co/microsoft/deberta-v3-small)',
      size: 0.578,
      f1: 0.5732,
    },
    {
      name: '[gliclass-small-v1.0](https://huggingface.co/knowledgator/gliclass-small-v1.0)',
      encoder: '[deberta-v3-small](https://huggingface.co/microsoft/deberta-v3-small)',
      size: 0.574,
      f1: 0.5401,
    },
    {
      name: '[gliclass-base-v1.0](https://huggingface.co/knowledgator/gliclass-base-v1.0)',
      encoder: '[deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base)',
      size: 0.745,
      f1: 0.5571,
    },
    {
      name: '[gliclass-large-v1.0](https://huggingface.co/knowledgator/gliclass-large-v1.0)',
      encoder: '[deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large)',
      size: 1.75,
      f1: 0.6078,
    },
    {
      name: '[gliclass-base-v2.0-rac-init](https://huggingface.co/knowledgator/gliclass-base-v2.0-rac-init)',
      encoder: '[deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base)',
      size: 0.745,
      f1: 0.5598,
    }
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
