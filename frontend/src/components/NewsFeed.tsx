import { useQuery } from '@tanstack/react-query';
import { newsApi } from '../services/api';
import type { NewsItem } from '../types';
import { Newspaper, ExternalLink, TrendingUp, TrendingDown, Minus, Clock } from 'lucide-react';

interface NewsFeedProps {
  symbol?: string;
  limit?: number;
  showFilters?: boolean;
}

export default function NewsFeed({ symbol, limit = 10, showFilters = true }: NewsFeedProps) {
  const { data, isLoading, error } = useQuery({
    queryKey: symbol ? ['news', symbol, limit] : ['news', limit],
    queryFn: () => 
      symbol 
        ? newsApi.getNewsBySymbol(symbol, limit).then(r => r.news)
        : newsApi.getNews({ limit }).then(r => r.news),
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 10 * 60 * 1000, // 10 minutes
  });

  if (isLoading) {
    return (
      <div className="bg-surface border border-border rounded-lg p-4">
        <div className="flex items-center gap-2 mb-4">
          <Newspaper className="w-5 h-5 text-primary" />
          <h2 className="text-lg font-semibold text-text">Crypto News</h2>
        </div>
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="animate-pulse bg-border/50 h-20 rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-surface border border-border rounded-lg p-4">
        <div className="flex items-center gap-2 mb-4">
          <Newspaper className="w-5 h-5 text-primary" />
          <h2 className="text-lg font-semibold text-text">Crypto News</h2>
        </div>
        <div className="text-danger text-sm">
          Failed to load news. Please try again later.
        </div>
      </div>
    );
  }

  const news = data || [];

  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Newspaper className="w-5 h-5 text-primary" />
          <h2 className="text-lg font-semibold text-text">
            {symbol ? `${symbol} News` : 'Crypto News'}
          </h2>
        </div>
        {showFilters && (
          <div className="flex gap-2">
            <span className="text-xs text-text-muted bg-background px-2 py-1 rounded">
              {news.length} articles
            </span>
          </div>
        )}
      </div>

      <div className="space-y-3 max-h-96 overflow-y-auto">
        {news.map((item: NewsItem) => (
          <NewsCard key={item.id} item={item} />
        ))}
      </div>
    </div>
  );
}

function NewsCard({ item }: { item: NewsItem }) {
  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return <TrendingUp className="w-4 h-4 text-success" />;
      case 'negative':
        return <TrendingDown className="w-4 h-4 text-danger" />;
      default:
        return <Minus className="w-4 h-4 text-text-muted" />;
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return 'bg-success/20 text-success border-success/30';
      case 'negative':
        return 'bg-danger/20 text-danger border-danger/30';
      default:
        return 'bg-text-muted/20 text-text-muted border-text-muted/30';
    }
  };

  const formatTimeAgo = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffHours < 1) return 'Just now';
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays === 1) return 'Yesterday';
    return `${diffDays} days ago`;
  };

  return (
    <div className="bg-background border border-border rounded-lg p-3 hover:border-primary/50 transition-colors">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-medium text-text line-clamp-2 mb-1">
            {item.title}
          </h3>
          <p className="text-xs text-text-muted line-clamp-2 mb-2">
            {item.summary}
          </p>
          
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-primary font-medium">
              {item.source}
            </span>
            <span className="text-text-muted">•</span>
            <span className="text-xs text-text-muted flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {formatTimeAgo(item.published_at)}
            </span>
            
            <span className={`text-xs px-2 py-0.5 rounded border ${getSentimentColor(item.sentiment)} flex items-center gap-1`}>
              {getSentimentIcon(item.sentiment)}
              {item.sentiment.charAt(0).toUpperCase() + item.sentiment.slice(1)}
            </span>
            
            <span className="text-xs text-text-muted bg-surface px-2 py-0.5 rounded">
              {item.category}
            </span>
          </div>
          
          {item.symbols && item.symbols.length > 0 && (
            <div className="flex gap-1 mt-2">
              {item.symbols.map((sym) => (
                <span key={sym} className="text-xs text-primary bg-primary/10 px-2 py-0.5 rounded">
                  {sym.replace('/USDT', '')}
                </span>
              ))}
            </div>
          )}
        </div>
        
        <a
          href={item.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-text-muted hover:text-primary transition-colors flex-shrink-0"
          title="Read full article"
        >
          <ExternalLink className="w-4 h-4" />
        </a>
      </div>
    </div>
  );
}
