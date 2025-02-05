import React, { useState } from 'react';
import { analyzeSentiments } from './sentimentService';
import styles from '../../styles/analysissentimenscss/AnalysisSentiment.module.css';

function AnalysisSentimentPage() {
  const [textInput, setTextInput] = useState('');  
  const [result, setResult] = useState([]);        
  const [loading, setLoading] = useState(false);   

  const handleSubmit = async (e) => {
    e.preventDefault();

    setLoading(true);

    try {
      const data = await analyzeSentiments([textInput]); 
      setResult(data.predictions);  
    } catch (error) {
      console.error('Erro durante a análise:', error);
      setResult([]);  
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles['analysis-container']}>
      <h1>Sentiment Analysis</h1>
      
      <form onSubmit={handleSubmit} className={styles.form}>
        <textarea
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
          placeholder="Type your phrases here"
          rows={6}
          className={styles['input-textarea']}
        />
        
        <button type="submit" disabled={loading} className={styles['submit-btn']}>
          {loading ? 'Analyzing...' : 'Analyze Sentiments'}
        </button>
      </form>

      <div className={styles['result-container']}>
        <h3>Results:</h3>
        {result.length > 0 ? (
          <ul className={styles['result-list']}>
            {result.map((sentiment, index) => (
              <li key={index} className={styles['result-item']}>
                {sentiment}
              </li>
            ))}
          </ul>
        ) : (
          <p>No results to display.</p>
        )}
      </div>
    </div>
  );
}

export default AnalysisSentimentPage;