/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        bg: 'rgb(var(--color-bg) / <alpha-value>)',
        text: 'rgb(var(--color-text) / <alpha-value>)',
        'sidebar-bg': 'rgb(var(--color-sidebar-bg) / <alpha-value>)',
        'card-bg': 'rgb(var(--color-card-bg) / <alpha-value>)',
        border: 'rgb(var(--color-border) / <alpha-value>)',
        'soft-bg': 'rgb(var(--color-soft-bg) / <alpha-value>)',
        'teal': {
          DEFAULT: '#008080',
          light: '#00a0a0',
          dark: '#006060',
        },
      },
      boxShadow: {
        'custom': '0 8px 20px rgba(0, 0, 0, 0.3)',
      },
    },
  },
  plugins: [],
}
