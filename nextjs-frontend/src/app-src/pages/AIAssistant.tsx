/**
 * AI Assistant Page
 * 
 * Replaces the Streamlit app on port 8502
 * Provides conversational AI interface for financial queries
 */

import { useState, useRef, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { portfolioApi, marketApi, ordersApi } from '../services/api';
import { Brain, Send, Bot, User, TrendingUp, Wallet, AlertCircle, Info, Loader2 } from 'lucide-react';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: string[];
}

const quickActions = [
  { label: 'Portfolio Status', prompt: 'Give me my current portfolio status', icon: Wallet },
  { label: 'Market Analysis', prompt: 'What is the current market sentiment?', icon: TrendingUp },
  { label: 'Risk Analysis', prompt: 'What are my current risk metrics?', icon: AlertCircle },
  { label: 'Recent Trades', prompt: 'Show my recent trades', icon: Info },
];

const financialTerms: Record<string, string> = {
  'sharpe ratio': 'A measure of risk-adjusted return. Higher is better. Values above 1.0 are considered good, above 2.0 excellent.',
  'var': 'Value at Risk - The maximum expected loss over a given time period at a given confidence level.',
  'cvar': 'Conditional Value at Risk - The expected loss beyond the VaR threshold.',
  'drawdown': 'The peak-to-trough decline in portfolio value before a new peak is reached.',
  'leverage': 'The use of borrowed money to increase potential returns.',
  'margin': 'Collateral deposited to cover credit risk.',
  'position': 'The amount of a security held by an investor.',
};

export default function AIAssistant() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      role: 'assistant',
      content: 'Hello! I\'m your AI Financial Assistant. Ask me about your portfolio, market conditions, risk metrics, or any financial terms. I can help you understand your trading performance and provide insights.',
      timestamp: new Date(),
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { data: summary } = useQuery({
    queryKey: ['portfolio-summary'],
    queryFn: portfolioApi.getSummary,
    refetchInterval: 30000,
  });

  const { data: positions } = useQuery({
    queryKey: ['portfolio-positions'],
    queryFn: () => portfolioApi.getPositions(),
    refetchInterval: 30000,
  });

  const { data: markets } = useQuery({
    queryKey: ['market-prices'],
    queryFn: marketApi.getAllPrices,
    refetchInterval: 30000,
  });

  const { data: orders } = useQuery({
    queryKey: ['orders'],
    queryFn: () => ordersApi.list(),
    refetchInterval: 30000,
  });

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const generateResponse = async (userInput: string): Promise<string> => {
    const input = userInput.toLowerCase();

    // Portfolio queries
    if (input.includes('portfolio') || input.includes('status') || input.includes('balance')) {
      const totalValue = (summary as any)?.total_value || 0;
      const pnl = (summary as any)?.daily_pnl || 0;
      const returnPct = (summary as any)?.daily_return_pct || 0;
      const numPositions = positions?.length || 0;
      
      return `📊 **Portfolio Status**\n\n` +
        `• Total Value: $${totalValue.toLocaleString()}\n` +
        `• Daily P/L: ${pnl >= 0 ? '+' : ''}$${pnl.toLocaleString()} (${returnPct.toFixed(2)}%)\n` +
        `• Open Positions: ${numPositions}\n\n` +
        `Would you like more details about your positions or performance?`;
    }

    // Market queries
    if (input.includes('market') || input.includes('price') || input.includes('btc') || input.includes('eth')) {
      const marketList = markets?.markets?.slice(0, 5).map((m: any) => 
        `• ${m.symbol}: $${m.price?.toLocaleString()} (${m.change_pct_24h >= 0 ? '+' : ''}${m.change_pct_24h?.toFixed(2)}%)`
      ).join('\n') || 'No market data available';
      
      return `📈 **Market Overview**\n\n${marketList}\n\n` +
        `Would you like detailed analysis on any specific asset?`;
    }

    // Trade queries
    if (input.includes('trade') || input.includes('order') || input.includes('recent')) {
      const recentOrders = orders?.slice(0, 5).map((o: any) => 
        `• ${o.side} ${o.quantity} ${o.symbol} @ $${o.price?.toFixed(2)} - ${o.status}`
      ).join('\n') || 'No recent orders';
      
      return `📋 **Recent Orders**\n\n${recentOrders}\n\n` +
        `Would you like to see more details or place a new order?`;
    }

    // Risk queries
    if (input.includes('risk') || input.includes('var') || input.includes('exposure')) {
      return `🛡️ **Risk Metrics**\n\n` +
        `• VaR (1-day, 95%): Calculated based on portfolio volatility\n` +
        `• Leverage: ${(summary as any)?.leverage || 1}x\n` +
        `• Margin Used: ${((summary as any)?.market_value || 0) / ((summary as any)?.total_value || 1) * 100}%\n\n` +
        `Would you like more detailed risk analysis?`;
    }

    // Financial term definitions
    for (const [term, definition] of Object.entries(financialTerms)) {
      if (input.includes(term)) {
        return `💡 **${term.toUpperCase()}**\n\n${definition}`;
      }
    }

    // Help
    if (input.includes('help') || input.includes('what can you')) {
      return `🤖 **I can help you with:**\n\n` +
        `• Portfolio status and performance\n` +
        `• Market prices and trends\n` +
        `• Recent trades and orders\n` +
        `• Risk metrics and analysis\n` +
        `• Financial term definitions\n` +
        `• Trading recommendations\n\n` +
        `Just ask me a question!`;
    }

    // Default response
    return `I understand you're asking about "${userInput}". \n\n` +
      `I can help with:\n` +
      `• Portfolio status - "What's my portfolio worth?"\n` +
      `• Market data - "How are BTC and ETH performing?"\n` +
      `• Recent trades - "Show my recent orders"\n` +
      `• Risk analysis - "What's my risk exposure?"\n\n` +
      `Or ask me to define any financial term!`;
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await generateResponse(input);
      
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickAction = (prompt: string) => {
    setInput(prompt);
  };

  return (
    <div className="flex h-[calc(100vh-100px)]">
      {/* Sidebar */}
      <div className="w-64 border-r border-white/10 p-4 hidden md:block">
        <div className="flex items-center gap-2 mb-6">
          <Brain className="w-6 h-6 text-primary" />
          <span className="font-bold text-text">Quick Actions</span>
        </div>
        <div className="space-y-2">
          {quickActions.map((action, index) => (
            <button
              key={index}
              onClick={() => handleQuickAction(action.prompt)}
              className="w-full flex items-center gap-3 px-4 py-3 rounded-lg bg-black/20 hover:bg-primary/20 text-text hover:text-primary transition-colors text-left"
            >
              <action.icon className="w-4 h-4" />
              <span className="text-sm">{action.label}</span>
            </button>
          ))}
        </div>

        <div className="mt-8">
          <h3 className="text-sm font-medium text-text-muted mb-3">Financial Terms</h3>
          <div className="space-y-1">
            {Object.keys(financialTerms).slice(0, 5).map(term => (
              <button
                key={term}
                onClick={() => setInput(`What is ${term}?`)}
                className="block text-sm text-text-muted hover:text-primary transition-colors"
              >
                {term}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-white/10">
          <div className="flex items-center gap-3">
            <Bot className="w-6 h-6 text-primary" />
            <div>
              <h1 className="text-xl font-bold text-text">AI Financial Assistant</h1>
              <p className="text-sm text-text-muted">Ask me anything about your portfolio</p>
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-lg p-4 ${
                  message.role === 'user'
                    ? 'bg-primary text-white'
                    : 'bg-black/30 text-text'
                }`}
              >
                <div className="flex items-start gap-2">
                  {message.role === 'assistant' && <Bot className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />}
                  {message.role === 'user' && <User className="w-5 h-5 flex-shrink-0 mt-0.5" />}
                  <div className="whitespace-pre-wrap text-sm">{message.content}</div>
                </div>
                <div className="text-xs text-text-muted mt-2">
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-black/30 rounded-lg p-4">
                <div className="flex items-center gap-2">
                  <Loader2 className="w-5 h-5 text-primary animate-spin" />
                  <span className="text-text-muted">Thinking...</span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="p-4 border-t border-white/10">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Ask about your portfolio, market, or trading..."
              className="flex-1 bg-black/30 border border-white/10 rounded-lg px-4 py-3 text-text placeholder-text-muted focus:outline-none focus:border-primary"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="px-6 py-3 bg-primary text-white rounded-lg hover:bg-primary/80 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
