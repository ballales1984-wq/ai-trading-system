/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      // Colors using CSS variables
      colors: {
        // Background colors
        background: 'var(--bg-primary)',
        'bg-primary': 'var(--bg-primary)',
        'bg-secondary': 'var(--bg-secondary)',
        'bg-tertiary': 'var(--bg-tertiary)',
        surface: 'var(--bg-secondary)',
        
        // Border colors  
        border: 'var(--border-color)',
        'border-hover': 'var(--border-hover)',
        
        // Text colors
        'text': 'var(--text-primary)',
        'text-primary': 'var(--text-primary)',
        'text-secondary': 'var(--text-secondary)',
        'text-muted': 'var(--text-muted)',
        
        // Accent colors
        primary: 'var(--accent-blue)',
        success: 'var(--accent-green)',
        danger: 'var(--accent-red)',
        warning: 'var(--accent-yellow)',
        purple: 'var(--accent-purple)',
        orange: 'var(--accent-orange)',
      },
      
      // Border radius
      borderRadius: {
        'card': '0.5rem',
      },
      
      // Animation
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'live-pulse': 'livePulse 2s ease-in-out infinite',
      },
      
      // Keyframes
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
      
      // Spacing
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
      },
      
      // Z-index
      zIndex: {
        '50': '50',
      }
    },
  },
  plugins: [],
}
