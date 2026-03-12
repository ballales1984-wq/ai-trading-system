import { Component, ErrorInfo, ReactNode } from 'react';
 
interface Props {
   children: ReactNode;
   fallbackRender?: (error: Error, resetError: () => void) => ReactNode;
 }
 
interface State {
   hasError: boolean;
   error?: Error;
   resetError: () => void;
 }
 
export class ErrorBoundary extends Component<Props, State> {
   constructor(props: Props) {
     super(props);
     this.state = { hasError: false, error: undefined, resetError: () => this.setState({ hasError: false, error: undefined }) };
   }
 
  static getDerivedStateFromError(error: Error): Partial<State> {
     return { hasError: true, error };
   }
 
   componentDidCatch(error: Error, errorInfo: ErrorInfo) {
     console.error('Error caught by boundary:', error, errorInfo);
     // You can also send error to monitoring service here
     // e.g., Sentry.captureException(error, errorInfo);
   }
 
   render() {
     if (this.state.hasError) {
       // If custom fallback render is provided, use it
       if (this.props.fallbackRender) {
         return this.props.fallbackRender(this.state.error!, this.state.resetError);
       }
 
       return (
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
                   onClick={this.state.resetError}
                   className="btn btn-outline w-full"
                 >
                   Try Again
                 </button>
               </div>
               {import.meta.env.DEV && this.state.error && (
                 <details className="mt-4 text-left">
                   <summary className="text-text-muted cursor-pointer">Error Details</summary>
                   <pre className="mt-2 p-2 bg-background rounded text-xs text-text-muted overflow-auto max-h-[200px]">
                     {this.state.error.stack}
                   </pre>
                 </details>
               )}
             </div>
           </div>
         </div>
       );
     }
 
     return this.props.children;
   }
 }
