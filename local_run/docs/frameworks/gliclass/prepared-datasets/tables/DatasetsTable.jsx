import React, { useState } from 'react';

export default function DatasetTable() {
  const [data, setData] = useState([
    {
      name: "[gliclass-v2.0](https://huggingface.co/datasets/gliclass-v2.0)",
      num_examples: 1196218,
      num_all_labels: 1382952,
      cache_files_size: 1.11
    },
    {
      name: "[gliclass-v2.0-RAC](https://huggingface.co/datasets/knowledgator/gliclass-v2.0-RAC)",
      num_examples: 612142,
      num_all_labels: 857027,
      cache_files_size: 1.31
    },
  ]);

  const [sortKey, setSortKey] = useState('num_examples');
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

  const renderLargeNumber = (num) => {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
  };  

  return (
    <table>
      <thead>
        <tr>
          {renderSortHeader('Name', 'name')}
          {renderSortHeader('Total examples', 'num_examples')}
          {renderSortHeader('Unique labels', 'num_all_labels')}
          {renderSortHeader('Cache size (GB)', 'cache_files_size')}
        </tr>
      </thead>
      <tbody>
        {sorted.map((row, index) => (
          <tr key={index}>
            <td>{renderMarkdownLink(row.name)}</td>
            <td>{renderLargeNumber(row.num_examples)}</td>
            <td>{renderLargeNumber(row.num_all_labels)}</td>
            <td>{row.cache_files_size}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
