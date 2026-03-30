import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'AI Trading System',
  description: 'Professional AI-powered trading system',
  icons: {
    icon: '/favicon.svg',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        {/* Google Analytics */}
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-K1XX9WGHL9"></script>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              window.dataLayer = window.dataLayer || [];
              function gtag(){dataLayer.push(arguments);}
              gtag('js', new Date());
              gtag('config', 'G-K1XX9WGHL9');
            `,
          }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
