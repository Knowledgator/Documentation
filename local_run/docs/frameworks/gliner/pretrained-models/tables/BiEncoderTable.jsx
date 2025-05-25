import React, { useState } from 'react';

export default function BiEncoderTable() {
  const [data, setData] = useState([
    {
      name: '[gliner-bi-small-v1.0](https://huggingface.co/knowledgator/gliner-bi-small-v1.0)',
      encoder: '[deberta-v3-small](https://huggingface.co/microsoft/deberta-v3-small)',
      labelEncoder: '[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)',
      size: 757,
      f1: 0.546,
    },
    {
      name: '[gliner-bi-base-v1.0](https://huggingface.co/knowledgator/gliner-bi-base-v1.0)',
      encoder: '[deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base)',
      labelEncoder: '[bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)',
      size: 969,
      f1: 0.563,
    },
    {
      name: '[gliner-bi-large-v1.0](https://huggingface.co/knowledgator/gliner-bi-large-v1.0)',
      encoder: '[deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large)',
      labelEncoder: '[bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)',
      size: 2280,
      f1: 0.567,
    },
    {
      name: '[gliner-poly-small-v1.0](https://huggingface.co/knowledgator/gliner-poly-small-v1.0)',
      encoder: '[deberta-v3-small](https://huggingface.co/microsoft/deberta-v3-small)',
      labelEncoder: '[bge-small-en-v1.5](https://huggingface.co/BBAAI/bge-small-en-v1.5)',
      size: 832,
      f1: 0.557,
    },
    {
      name: '[gliner-bi-llama-v1.0](https://huggingface.co/knowledgator/gliner-bi-llama-v1.0)',
      encoder: '[Sheared-LLaMA-encoder-1.3B](https://huggingface.co/knowledgator/Sheared-LLaMA-encoder-1.3B)',
      labelEncoder: '[bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)',
      size: 5440,
      f1: 0.589,
    },
    {
      name: '[modern-gliner-bi-base-v1.0](https://huggingface.co/knowledgator/modern-gliner-bi-base-v1.0)',
      encoder: '[ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)',
      labelEncoder: '[bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)',
      size: 830,
      f1: 0.594,
    },
    {
      name: '[modern-gliner-bi-large-v1.0](https://huggingface.co/knowledgator/modern-gliner-bi-base-v1.0)',
      encoder: '[ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large)',
      labelEncoder: '[bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)',
      size: 2120,
      f1: 0.598,
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
          {renderSortHeader('Labels Encoder', 'labelEncoder')}
          {renderSortHeader('Size (MB)', 'size')}
          {renderSortHeader('Zero-Shot F1 Score', 'f1')}
        </tr>
      </thead>
      <tbody>
        {sorted.map((row, index) => (
          <tr key={index}>
            <td>{renderMarkdownLink(row.name)}</td>
            <td>{renderMarkdownLink(row.encoder)}</td>
            <td>{renderMarkdownLink(row.labelEncoder)}</td>
            <td>{row.size}</td>
            <td>{row.f1}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
