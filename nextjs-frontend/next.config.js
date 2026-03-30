/** @type {import('next').NextConfig} */
const nextConfig = {
  // Proxy /api/* to the backend (set NEXT_PUBLIC_API_URL env var in Kilo Deploy)
  async rewrites() {
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/api/:path*`,
      },
      {
        source: '/ws/:path*',
        destination: `${backendUrl}/ws/:path*`,
      },
    ];
  },
  // Suppress punycode deprecation warning from dependencies
  webpack: (config) => {
    config.resolve.fallback = { ...config.resolve.fallback, punycode: false };
    // Resolve @/ alias to src/app-src/ (where the React SPA source lives)
    config.resolve.alias = {
      ...config.resolve.alias,
      '@': require('path').resolve(__dirname, 'src/app-src'),
    };
    return config;
  },
};

module.exports = nextConfig;
