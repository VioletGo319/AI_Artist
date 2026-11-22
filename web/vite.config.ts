import { resolve } from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig(({ command }) => ({
  base: './',
  plugins: [react(), tailwindcss()],
  publicDir: command === 'serve' && process.env.ART_DEMO_PUBLIC
    ? resolve(process.env.ART_DEMO_PUBLIC) : 'public',
}))
