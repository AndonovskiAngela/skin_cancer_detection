import React, { useState } from 'react';
import Modal from 'react-modal';
import axios from 'axios';
import './App.css';

Modal.setAppElement('#root');

const customStyles = {
  content: {
    top: '50%',
    left: '50%',
    right: 'auto',
    bottom: 'auto',
    marginRight: '-50%',
    transform: 'translate(-50%, -50%)',
    backgroundColor: '#f0f0f0', // Light grey background
    borderRadius: '10px',
    padding: '20px',
    borderColor: '#ccc',
  },
};

function App() {
  const [modalIsOpen, setModalIsOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState("");

  const openModal = () => setModalIsOpen(true);
  const closeModal = () => setModalIsOpen(false);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      alert("Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://localhost:5000/test', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPrediction(response.data);
    } catch (error) {
      console.error("Error uploading image:", error);
      setPrediction("Error predicting the result. Please try again.");
    }

    closeModal();
  };

  return (
    <div className="App">
      <div style={{ backgroundColor: '#007bff', color: 'white', padding: '10px 0', textAlign: 'center', marginBottom: '20px' }}>
        Skin Cancer Detection
      </div>
      <button onClick={openModal} style={{ position: 'absolute', left:'45%', top:'15%', padding: '10px 20px', fontSize: '16px', borderRadius: '5px', cursor: 'pointer', backgroundColor: '#007bff', color: 'white', border: 'none' }}>
        Upload Image
      </button>
      <Modal
        isOpen={modalIsOpen}
        onRequestClose={closeModal}
        style={customStyles}
        contentLabel="Upload Image"
      >
        <h2>Upload Image</h2>
        <input type="file" onChange={handleFileChange} />
        <div style={{ marginTop: '20px' }}>
          <button onClick={handleSubmit} style={{ marginRight: '10px', padding: '10px 20px', fontSize: '16px', borderRadius: '5px', cursor: 'pointer', backgroundColor: '#28a745', color: 'white', border: 'none' }}>Submit</button>
          <button onClick={closeModal} style={{ padding: '10px 20px', fontSize: '16px', borderRadius: '5px', cursor: 'pointer', backgroundColor: '#6c757d', color: 'white', border: 'none' }}>Close</button>
        </div>
      </Modal>
      {prediction && <div style={{position: 'absolute', left:'38%', top:'10%' }}>Prediction: <strong>{prediction.class}</strong> with an accuracy of: <strong>{prediction.confidence}%</strong></div>}
    </div>
  );
}

export default App;
