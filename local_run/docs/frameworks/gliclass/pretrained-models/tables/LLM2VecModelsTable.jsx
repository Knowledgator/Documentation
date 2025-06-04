import React, { useState } from 'react';

export default function BiEncoderTable() {
  const [data, setData] = useState([
    {
      name: '[gliclass-llama-1.3B-v1.0](https://huggingface.co/knowledgator/gliclass-llama-1.3B-v1.0)',
      encoder: '[Sheared-LLaMA-encoder-1.3B](https://huggingface.co/knowledgator/Sheared-LLaMA-encoder-1.3B)',
      size: 5.182, 
      f1: 0.6927,
    },
    {
      name: '[gliclass-qwen-0.5B-v1.0](https://huggingface.co/knowledgator/gliclass-qwen-0.5B-v1.0)',
      encoder: '[Qwen-encoder-0.5B](https://huggingface.co/knowledgator/Qwen-encoder-0.5B)',
      size: 1.99,
      f1: 0.6445,
    },
    {
      name: '[gliclass-qwen-1.5B-v1.0](https://huggingface.co/knowledgator/gliclass-qwen-1.5B-v1.0)',
      encoder: '[Qwen-encoder-1.5B](https://huggingface.co/knowledgator/Qwen-encoder-1.5B)',
      size: 6.21,
      f1: 0.6956,
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
