import ClientApp from '../ClientApp';

// Catch-all route: lets BrowserRouter in the React SPA handle all client-side routing
export default function CatchAllPage() {
  return <ClientApp />;
}
