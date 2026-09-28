/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        tactical: {
          bg: '#090A0F',
          surface: '#12141A',
          card: '#1C202A',
          border: '#2A303F',
          muted: '#64748B',
          text: '#F1F5F9',
          primary: '#38BDF8',   // Steel Blue
          success: '#10B981',   // Emerald
          warning: '#F59E0B',   // Tactical Amber
          danger: '#EF4444',    // Crimson Threat
        }
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'monospace'],
        sans: ['Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
