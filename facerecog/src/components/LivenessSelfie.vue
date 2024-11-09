<template>
    <v-container>
      <v-btn @click="takeSelfie">Take Liveness Selfie</v-btn>
    </v-container>
  </template>
  
  <script>
  import { compareImages } from '@/composables/apiService.js';
  
  export default {
    methods: {
      async takeSelfie() {
        // Simulate capturing a selfie
        const selfie = new File([""], "selfie.jpg"); // Replace with actual selfie capture logic
  
        try {
          const response = await compareImages(selfie);
          console.log('Comparison response:', response.data);
        } catch (error) {
          console.error('Error comparing images:', error);
        }
      }
    }
  }
  </script>