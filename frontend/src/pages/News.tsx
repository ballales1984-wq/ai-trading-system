import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { newsApi } from '../services/api';
import { RefreshCw, Newspaper, TrendingUp, TrendingDown, Minus, Loader2 } from 'lucide-react';

export default function News() {
  const [filter, setFilter] = useState<'all' | 'positive' | 'negative'>('all');
  
  const { data, isLoading, isFetching, refetch } = useQuery({
    queryKey: ['news'],
    queryFn: () => newsApi.getNews({ limit: 50 }),
  });

  const newsItems = data?.news || [];
  
  const filteredNews = filter === 'all' 
    ? newsItems 
    : newsItems.filter((item: any) => item.sentiment === filter);

  const getSentimentIcon = (sentiment?: string) => {
    switch (sentiment) {
      case 'positive': return <TrendingUp size={14} className="text-success" />;
      case 'negative': return <TrendingDown size={14} className="text-danger" />;
      default: return <Minus size={14} className="text-text-muted" />;
    }
  };

  const getSentimentBg = (sentiment?: string) => {
    switch (sentiment) {
      case 'positive': return 'bg-success/10 border-success/30';
      case 'negative': return 'bg-danger/10 border-danger/30';
      default: return 'bg-white/5 border-white/10';
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 className="w-8 h-8 text-primary animate-spin" />
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text">Market Intelligence</h1>
          <p className="text-text-muted">Real-time financial news and sentiment analysis</p>
        </div>
        
        <div className="flex items-center gap-3 w-full md:w-auto">
          <div className="flex items-center gap-1 bg-white/5 p-1 rounded-xl border border-white/10">
            <FilterButton active={filter === 'all'} label="All" onClick={() => setFilter('all')} />
            <FilterButton active={filter === 'positive'} label="Bullish" onClick={() => setFilter('positive')} />
            <FilterButton active={filter === 'negative'} label="Bearish" onClick={() => setFilter('negative')} />
          </div>
          
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="flex items-center justify-center p-2.5 bg-primary/10 text-primary border border-primary/30 rounded-xl hover:bg-primary/20 transition-all font-semibold"
            title="Refresh News Feed"
          >
            <RefreshCw size={18} className={isFetching ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* News Feed */}
      <div className="grid grid-cols-1 gap-4">
        {filteredNews.length === 0 ? (
          <div className="premium-glass-panel py-20 text-center text-text-muted">
            <Newspaper className="mx-auto mb-4 opacity-20" size={48} />
            <p>No intelligence reports found for the selected filter.</p>
          </div>
        ) : (
          filteredNews.map((item: any) => (
            <div
              key={item.id}
              className="premium-glass-panel p-5 premium-glass-hover border-white/[0.05] flex flex-col md:flex-row gap-4"
            >
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider border flex items-center gap-1.5 ${getSentimentBg(item.sentiment)}`}>
                    {getSentimentIcon(item.sentiment)}
                    {item.sentiment || 'neutral'}
                  </span>
                  <span className="text-xs text-text-muted font-mono">{new Date(item.published_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                </div>
                <h3 className="text-lg font-bold text-text mb-2 leading-tight">
                  {item.title}
                </h3>
                <p className="text-sm text-text-muted leading-relaxed line-clamp-3 md:line-clamp-2">
                  {item.content}
                </p>
              </div>
              
              <div className="flex flex-col justify-between items-end shrink-0 pt-2 md:pt-0">
                <div className="flex gap-1.5 flex-wrap justify-end">
                  {item.symbols?.map((symbol: string) => (
                    <span
                      key={symbol}
                      className="px-2 py-0.5 bg-primary/10 text-primary border border-primary/20 rounded text-[10px] font-bold"
                    >
                      {symbol}
                    </span>
                  ))}
                </div>
                <span className="text-xs text-text-muted font-semibold mt-4">Source: {item.source}</span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function FilterButton({ active, label, onClick }: any) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
        active 
          ? 'bg-primary text-white shadow-lg glow-primary' 
          : 'text-text-muted hover:text-text hover:bg-white/5'
      }`}
    >
      {label}
    </button>
  );
}

