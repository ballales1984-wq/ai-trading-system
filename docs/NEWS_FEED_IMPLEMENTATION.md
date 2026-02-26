# 📰 News Feed Implementation

## Overview

Real-time crypto news feed with sentiment analysis integrated into the Market page.

## Features

- **Live News Feed**: Latest crypto news from multiple sources
- **Sentiment Analysis**: Positive/Negative/Neutral indicators with visual badges
- **Symbol Tagging**: News items tagged with relevant trading symbols
- **Auto-refresh**: Updates every 10 minutes
- **Responsive Design**: Mobile-friendly card layout

## API Endpoints

### Backend

```text
GET /api/v1/news              # Get all news
GET /api/v1/news/{symbol}     # Get news for specific symbol
```

### Frontend Types

```typescript
interface NewsItem {
  id: string;
  title: string;
  source: string;
  url: string;
  summary: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  sentiment_score: number;
  symbols: string[];
  published_at: string;
  category: string;
}
```

## Components

### NewsFeed.tsx

Main component with:

- Loading skeletons
- Error handling
- Time-ago formatting
- Sentiment badges with icons
- Symbol tags
- External link to full article

## Integration

Integrated into `Market.tsx` page below the charts section.

## Mock Data

10 sample news articles covering:

- Bitcoin ETF approvals
- Ethereum upgrades
- Solana ecosystem
- Regulatory news
- Market analysis

## Styling

- Dark theme matching dashboard
- Color-coded sentiment badges
- Hover effects on cards
- Scrollable container (max-height: 24rem)
