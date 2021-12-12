import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path';

// https://vitejs.dev/config/
export default defineConfig(({mode}) => {
  return {
    plugins: [react({
      babel: {
        plugins: [
          [
            'babel-plugin-styled-components',
            {fileName: false, displayName: mode === 'development'},
          ],
        ],
      },
    })],
    build: {
      rollupOptions: {
        input: {
          main: resolve(__dirname, 'index.html'),
          arcade: resolve(__dirname, 'arcade/index.html')
        }
      }
    },
  };
})
