import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import AnalysisSentiment from './pages/analysissentiment/AnalysisSentimentPage'; 

function App() {
  return (
    <Router>
          <Routes>
            <Route path="/" element={<AnalysisSentiment />} />
          </Routes>
    </Router>
  );
}

export default App;