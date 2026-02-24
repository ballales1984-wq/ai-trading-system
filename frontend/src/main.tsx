import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from './App'
import { ErrorBoundary } from './components/ui/ErrorBoundary'
import { sendClientEvent } from './utils/telemetry'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 30000,
    },
  },
})

window.addEventListener('error', (event) => {
  void sendClientEvent({
    level: 'error',
    event: 'window_error',
    details: { message: event.message, source: event.filename, line: event.lineno },
  });
});

window.addEventListener('unhandledrejection', (event) => {
  void sendClientEvent({
    level: 'error',
    event: 'unhandled_rejection',
    details: { reason: String(event.reason) },
  });
});

const root = createRoot(document.getElementById('root')!);
root.render(
  <ErrorBoundary>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </ErrorBoundary>
);


