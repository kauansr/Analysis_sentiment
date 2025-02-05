import axios from 'axios';

export const analyzeSentiments = async (sentences) => {
  try {
    const response = await axios.post('http://localhost:8000/predict', {
      texto: sentences, 
    });
    return response.data;
  } catch (error) {
    console.error('Erro ao enviar para o backend:', error);
    throw error; 
  }
};