export function DemoBadge() {
  return (
    <div className="fixed top-4 right-4 z-50">
      <div className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-4 py-2 rounded-full text-sm font-medium shadow-lg animate-pulse-slow flex items-center gap-2">
        <span className="w-2 h-2 bg-white rounded-full animate-pulse"></span>
        Demo Mode
      </div>
    </div>
  );
}

export function DemoBanner() {
  return (
    <div className="bg-gradient-to-r from-purple-600/20 to-pink-600/20 border-b border-purple-500/30 py-2 px-4 text-center">
      <p className="text-sm text-purple-200">
        🎮 <span className="font-medium">Demo Mode</span> - Using simulated data. 
        <a href="/landing" className="ml-2 text-purple-400 hover:text-purple-300 underline">
          Join waitlist for real trading
        </a>
      </p>
    </div>
  );
}
