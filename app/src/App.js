import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

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
      <div className="app-container">
        <div className="content-wrapper">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
            placeholder="Enter your question..."
            className="input-field"
          />
          <button
            onClick={handleSubmit}
            className="submit-button"
          >
            Submit
          </button>
        </div>

        {aiResponse && (
          <div className="response-container">
            <ReactMarkdown>{aiResponse}</ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
