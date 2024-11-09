<!-- frontend/src/App.vue -->
<template>
  <v-app>
    <v-main>
      <file-upload />
      <liveness-selfie />
      <validation-result />
    </v-main>
  </v-app>
</template>

<script>
import FileUpload from './components/FileUpload.vue';
import LivenessSelfie from './components/LivenessSelfie.vue';
import ValidationResult from './components/ValidationResult.vue';

export default {
  components: {
    FileUpload,
    LivenessSelfie,
    ValidationResult
  }
}
</script>