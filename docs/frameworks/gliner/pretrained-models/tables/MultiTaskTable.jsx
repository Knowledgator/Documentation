import React, { useState } from 'react';

export default function BiEncoderTable() {
  const [data, setData] = useState([
    {
      name: '[gliner-multitask-large-v0.5](https://huggingface.co/knowledgator/gliner-multitask-large-v0.5)',
      encoder: '[deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large)',
      size: 1.76,
      f1: 0.6276,
    },
    {
      name: '[gliner-multitask-v1.0](https://huggingface.co/knowledgator/gliner-multitask-v1.0)',
      encoder: '[deberta-v2-xlarge](https://huggingface.co/microsoft/deberta-v2-xlarge)',
      size: 3.64,
      f1: 0.6325,
     },
     {
      name: '[gliner-llama-multitask-1B-v1.0](https://huggingface.co/knowledgator/gliner-llama-multitask-1B-v1.0)',
      encoder: '[Llama-encoder-1.0B](https://huggingface.co/knowledgator/Llama-encoder-1.0B)',
      size: 4.24,
      f1: 0.6153,
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
