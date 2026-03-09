import { useState, useEffect } from 'react';
import { api } from '../services/api';

interface NewsItem {
  id: number;
  title: string;
  content: string;
  source: string;
  sentiment?: 'positive' | 'negative' | 'neutral';
  published_at: string;
  symbols?: string[];
}

const News = () => {
  const [news, setNews] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'positive' | 'negative'>('all');

  useEffect(() => {
    fetchNews();
  }, []);

  const fetchNews = async () => {
    try {
      setLoading(true);
      const response = await api.get('/api/news');
      setNews(response.data);
    } catch (err) {
      // Demo data fallback
      setNews([
        {
          id: 1,
          title: 'Market Update: BTC Reaches New Highs',
          content: 'Bitcoin has surged past previous resistance levels...',
          source: 'CryptoNews',
          sentiment: 'positive',
          published_at: new Date().toISOString(),
          symbols: ['BTC', 'ETH']
        },
        {
          id: 2,
          title: 'Fed Announces Interest Rate Decision',
          content: 'Federal Reserve maintains current interest rates...',
          source: 'Financial Times',
          sentiment: 'neutral',
          published_at: new Date().toISOString(),
          symbols: ['SPY', 'QQQ']
        },
        {
          id: 3,
          title: 'Tech Stocks Face Pressure',
          content: 'Major tech companies report earnings below expectations...',
          source: 'Bloomberg',
          sentiment: 'negative',
          published_at: new Date().toISOString(),
          symbols: ['AAPL', 'MSFT', 'GOOGL']
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const filteredNews = filter === 'all' 
    ? news 
    : news.filter(item => item.sentiment === filter);

  const getSentimentColor = (sentiment?: string) => {
    switch (sentiment) {
      case 'positive': return 'text-green-500';
      case 'negative': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getSentimentBg = (sentiment?: string) => {
    switch (sentiment) {
      case 'positive': return 'bg-green-500/10 border-green-500/20';
      case 'negative': return 'bg-red-500/10 border-red-500/20';
      default: return 'bg-gray-500/10 border-gray-500/20';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-white">Market News</h1>
          <p className="text-gray-400">Latest financial news and sentiment analysis</p>
        </div>
        
        {/* Filter Buttons */}
        <div className="flex gap-2">
          <button
            onClick={() => setFilter('all')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              filter === 'all'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            All
          </button>
          <button
            onClick={() => setFilter('positive')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              filter === 'positive'
                ? 'bg-green-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Positive
          </button>
          <button
            onClick={() => setFilter('negative')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              filter === 'negative'
                ? 'bg-red-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Negative
          </button>
        </div>
      </div>

      {/* News Feed */}
      <div className="space-y-4">
        {filteredNews.length === 0 ? (
          <div className="text-center py-12 text-gray-400">
            No news available
          </div>
        ) : (
          filteredNews.map((item) => (
            <div
              key={item.id}
              className={`p-4 rounded-lg border ${getSentimentBg(item.sentiment)} backdrop-blur-sm`}
            >
              <div className="flex justify-between items-start mb-2">
