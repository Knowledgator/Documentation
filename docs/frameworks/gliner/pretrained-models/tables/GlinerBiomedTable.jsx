import React, { useState } from 'react';

export default function GlinerBiomedTable() {
  const [data, setData] = useState([
    {
      name: '[gliner-biomed-small-v1.0](https://huggingface.co/Ihor/gliner-biomed-small-v1.0)',
      encoder: '[deberta-v3-small](https://huggingface.co/microsoft/deberta-v3-small)',
      labelEncoder: "-",
      size: 611,
      f1: 0.5253,
    },
    {
      name: '[gliner-biomed-base-v1.0](https://huggingface.co/Ihor/gliner-biomed-base-v1.0)',
      encoder: '[deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base)',
      labelEncoder: "-",
      size: 781,
      f1: 0.5437,
    },
    {
      name: '[gliner-biomed-large-v1.0](https://huggingface.co/Ihor/gliner-biomed-large-v1.0)',
      encoder: '[deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large)',
      labelEncoder: "-",
      size: 781,
      f1: 0.5977,
    },
    {
      name: '[gliner-biomed-bi-small-v1.0](https://huggingface.co/Ihor/gliner-biomed-bi-small-v1.0)',
      encoder: '[deberta-v3-small](https://huggingface.co/microsoft/deberta-v3-small)',
      labelEncoder: '[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)',
      size: 757,
      f1: 0.5693,
    },
    {
      name: '[gliner-biomed-bi-base-v1.0](https://huggingface.co/Ihor/gliner-biomed-bi-base-v1.0)',
      encoder: '[deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base)',
      labelEncoder: '[bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)',
      size: 969,
      f1: 0.5831,
    },
    {
      name: '[gliner-biomed-bi-large-v1.0](https://huggingface.co/Ihor/gliner-biomed-bi-large-v1.0)',
      encoder: '[deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large)',
      labelEncoder: '[bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)',
      size: 2334,
      f1: 0.5490,
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
          {renderSortHeader('Size MB', 'size')}
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
