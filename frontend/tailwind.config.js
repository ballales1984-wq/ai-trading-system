/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      // Colors using CSS variables for theming and future dark/light mode support
      colors: {
        // Background colors
        'bg-primary': 'var(--bg-primary)',
        'bg-secondary': 'var(--bg-secondary)',
        background: 'var(--bg-primary)',
        surface: 'var(--bg-secondary)',
        
        // Border color
        border: 'var(--border-color)',
        
        // Text colors
        'text-primary': 'var(--text-primary)',
        'text-secondary': 'var(--text-secondary)',
        text: 'var(--text-primary)',
        'text-muted': 'var(--text-secondary)',
        
        // Accent colors (semantic naming)
        primary: 'var(--accent-blue)',
        success: 'var(--accent-green)',
        danger: 'var(--accent-red)',
        warning: 'var(--accent-yellow)',
        
        // Direct accent color aliases for flexibility
        'accent-blue': 'var(--accent-blue)',
        'accent-green': 'var(--accent-green)',
        'accent-red': 'var(--accent-red)',
        'accent-yellow': 'var(--accent-yellow)',
      },
      
      // Border radius extensions
      borderRadius: {
        'card': '0.5rem',
      },
      
      // Animation extensions
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'live-pulse': 'livePulse 2s ease-in-out infinite',
      },
      
      // Keyframes for custom animations
      keyframes: {
        livePulse: {
          '0%, 100%': {
            opacity: '1',
            boxShadow: '0 0 0 0 rgba(63, 185, 80, 0.7)',
          },
          '50%': {
            opacity: '0.7',
            boxShadow: '0 0 0 8px rgba(63, 185, 80, 0)',
          },
        },
      },
    },
  },
  plugins: [],
}

