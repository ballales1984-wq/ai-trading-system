import { createRoot } from 'react-dom/client'
  import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
  import App from './App'
  import { ErrorBoundary } from './components/ui/ErrorBoundary'
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
 
  const root = createRoot(document.getElementById('root')!);
  root.render(
    <ErrorBoundary fallbackRender={(_, resetError) => (
      <div className="min-h-screen bg-background flex items-center justify-center p-4">
        <div className="bg-surface border border-border rounded-lg p-6 max-w-md w-full">
          <div className="text-center">
            <div className="w-16 h-16 bg-danger/20 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-3xl">⚠️</span>
            </div>
            <h2 className="text-xl font-bold text-text mb-2">Something went wrong</h2>
            <p className="text-text-muted mb-4">
              The application encountered an error. Please try refreshing the page.
            </p>
            <div className="flex flex-col space-y-3">
              <button
                onClick={() => window.location.reload()}
                className="btn btn-primary w-full"
              >
                Refresh Page
              </button>
              <button
                onClick={resetError}
                className="btn btn-outline w-full"
              >
                Try Again
              </button>
            </div>
          </div>
        </div>
      </div>
    )}>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </ErrorBoundary>
  );


