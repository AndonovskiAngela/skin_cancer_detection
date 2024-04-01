import React, { useState } from 'react';
import { Button, Modal, Box, Typography } from '@mui/material';
import axios from 'axios';

const style = {
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: 400,
  bgcolor: 'background.paper',
  border: '2px solid #000',
  boxShadow: 24,
  p: 4,
};

function ImageUploadModal({ open, handleClose, setResult }) {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleSubmit = async () => {
    if (selectedFile) {
      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const response = await axios.post('http://localhost:5000/test', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        setResult(response.data); // Assuming the API returns the string directly
        handleClose(); // Close modal after submission
      } catch (error) {
        console.error('Error uploading image:', error);
        setResult('Error uploading image');
      }
    }
  };

  return (
    <Modal
      open={open}
      onClose={handleClose}
      aria-labelledby="modal-modal-title"
      aria-describedby="modal-modal-description"
    >
      <Box sx={style}>
        <Typography id="modal-modal-title" variant="h6" component="h2">
          Upload a Skin Image
        </Typography>
        <input type="file" onChange={handleFileChange} />
        <Button onClick={handleSubmit}>Submit</Button>
      </Box>
    </Modal>
  );
}

export default ImageUploadModal;
