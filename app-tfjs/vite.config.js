import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          ui: ["react", "react-dom", "@mantine/core", "@mantine/hooks"],
          widgets: ["@tabler/icons-react", "recharts"],
          ml: ["@tensorflow/tfjs"],
        }
      }
    }
  }
})
