/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './src/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        background: 'hsl(var(--background))',
        surface: 'hsl(var(--surface))',
        'surface-glass': 'hsl(var(--surface-glass))',
        border: 'hsl(var(--border))',
        'border-highlight': 'hsl(var(--border-highlight))',

        text: 'hsl(var(--text-primary))',
        'text-primary': 'hsl(var(--text-primary))',
        'text-secondary': 'hsl(var(--text-secondary))',
        'text-muted': 'hsl(var(--text-muted))',

        primary: 'hsl(var(--primary))',
        success: 'hsl(var(--success))',
        danger: 'hsl(var(--danger))',
        warning: 'hsl(var(--warning))',
        purple: 'hsl(var(--purple))',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'live-pulse': 'livePulse 2s ease-in-out infinite',
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
};
