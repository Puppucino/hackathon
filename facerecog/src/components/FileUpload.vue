<template>
    <v-container>
      <v-file-input label="Upload ID Image" @change="onFileChange" />
      <v-btn @click="captureImage">Capture with Camera</v-btn>
    </v-container>
  </template>
  
  <script>
  import { uploadImage } from '@/composables/apiService.js';
  
  export default {
    methods: {
      async onFileChange(file) {
        try {
          const response = await uploadImage(file);
          console.log('Upload response:', response.data);
        } catch (error) {
          console.error('Error uploading image:', error);
        }
      },
      captureImage() {
        // Handle camera capture
        console.log('Capture image with camera');
      }
    }
  }
  </script>