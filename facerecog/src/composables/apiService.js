// frontend/src/apiService.js
import axios from 'axios';

const API_URL = 'http://127.0.0.1:5000'; // Ensure this matches your Flask server URL

export const uploadImage = (file) => {
  const formData = new FormData();
  formData.append('file', file);

  return axios.post(`${API_URL}/upload`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

export const compareImages = (selfie) => {
  const formData = new FormData();
  formData.append('selfie', selfie);

  return axios.post(`${API_URL}/compare`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};