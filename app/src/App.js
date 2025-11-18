import './App.css';
import { useState } from 'react';
import ReactMarkdown from 'react-markdown';

function App() {
  const [inputText, setInputText] = useState('');
  const [aiResponse, setAiResponse] = useState('');

  const handleSubmit = async () => {
    if (!inputText.trim()) return;

    try {
      setAiResponse('Loading...');
      const response = await fetch(
        `http://localhost:8000/retrieve-answers?query_text=${encodeURIComponent(inputText)}`
      );

      if (!response.ok) {
        throw new Error('API request failed');
      }

      const data = await response.json();

      // Extract the markdown response from the API
      if (data.response && data.response.message) {
        setAiResponse(data.response.message);
      } else {
        setAiResponse('No response received from the API');
      }
    } catch (error) {
      console.error('Error fetching response:', error);
      setAiResponse(`**Error:** ${error.message}\n\nMake sure the API server is running on port 8000.`);
    }
  };

  return (
    <div className="App">
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        padding: '20px'
      }}>
        <div style={{
          maxWidth: '800px',
          width: '100%',
          textAlign: 'center'
        }}>
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
            placeholder="Enter your question..."
            style={{
              width: '100%',
              padding: '12px',
              fontSize: '16px',
              borderRadius: '8px',
              border: '2px solid #ddd',
              marginBottom: '20px',
              boxSizing: 'border-box'
            }}
          />
          <button
            onClick={handleSubmit}
            style={{
              padding: '12px 24px',
              fontSize: '16px',
              borderRadius: '8px',
              border: 'none',
              backgroundColor: '#007bff',
              color: 'white',
              cursor: 'pointer',
              marginBottom: '40px'
            }}
          >
            Submit
          </button>
        </div>

        {aiResponse && (
          <div style={{
            maxWidth: '800px',
            width: '100%',
            textAlign: 'left',
            padding: '20px',
            backgroundColor: '#f5f5f5',
            borderRadius: '8px'
          }}>
            <ReactMarkdown>{aiResponse}</ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
