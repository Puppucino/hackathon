<!-- frontend/src/components/ValidationResult.vue -->
<template>
    <v-container>
      <div v-if="isValidated">User is validated!</div>
      <div v-else>User validation failed.</div>
    </v-container>
  </template>
  
  <script>
  export default {
    data() {
      return {
        isValidated: false // This should be set based on backend response
      };
    }
  }
  </script>