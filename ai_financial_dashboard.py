"""
AI Financial Assistant Dashboard - Streamlit
============================================
Dashboard interattivo per l'analisi del portafoglio finanziario.
Supporta domande in linguaggio naturale e aggiornamento automatico dei dati.

Autore: AI Trading System
Data: 2026-03-17
"""

import streamlit as st
import requests
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configurazione pagina
st.set_page_config(
    page_title="AI Financial Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurazione API
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# Colori per il tema
COLORS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "warning": "#f59e0b",
    "neutral": "#6b7280",
    "background": "#0f172a",
    "card_bg": "#1e293b"
}

# ==================== DATA FETCHING ====================

@st.cache_data(ttl=30)
def fetch_portfolio_summary() -> Dict:
    """Recupera dati portafoglio dall'API"""
    try:
        response = requests.get(f"{API_BASE_URL}/portfolio/summary", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Errore API: {e}")
    return {}

@st.cache_data(ttl=30)
def fetch_positions() -> List[Dict]:
    """Recupera posizioni dall'API"""
    try:
        response = requests.get(f"{API_BASE_URL}/portfolio/positions", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Errore API: {e}")
    return []

@st.cache_data(ttl=60)
def fetch_news() -> Dict:
    """Recupera notizie dall'API"""
    try:
        response = requests.get(f"{API_BASE_URL}/news", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Errore API: {e}")
    return {}

# ==================== VOCABULARY MANAGER ====================

class FinancialVocabulary:
    """Gestisce il vocabolario finanziario del bot"""
    
    def __init__(self):
        self.terms = {
            # Termini base crypto
            "spot_etf": {
                "term": "Spot ETF",
                "definition": "Exchange Traded Fund che detiene direttamente l'asset sottostante (Bitcoin/Ethereum)",
                "category": "product"
            },
            "staking_yields": {
                "term": "Staking Yields",
                "definition": "Rendimenti ottenuti bloccando token per partecipare alla validazione della blockchain",
                "category": "defi"
            },
            "layer2": {
                "term": "Layer 2 (L2)",
                "definition": "Soluzioni di scalabilità costruite su blockchain principali (Arbitrum, Optimism, Base)",
                "category": "technology"
            },
            "impermanent_loss": {
                "term": "Impermanent Loss",
                "definition": "Perdita temporanea di valore nei liquidity pool quando i prezzi degli asset divergono",
                "category": "defi"
            },
            "trailing_stop": {
                "term": "Trailing Stop",
                "definition": "Stop-loss dinamico che si muove con il prezzo per proteggere i profitti",
                "category": "trading"
            },
            "var": {
                "term": "VaR (Value at Risk)",
                "definition": "Massima perdita attesa in un intervallo di confidenza (es. 95%)",
                "category": "risk"
            },
            "cvar": {
                "term": "CVaR (Conditional VaR)",
                "definition": "Perdita attesa condizionata quando si supera il VaR (Expected Shortfall)",
                "category": "risk"
            },
            "sharpe_ratio": {
                "term": "Sharpe Ratio",
                "definition": "Misura del rendimento aggiustato per il rischio (ritorno - risk-free / volatilità)",
                "category": "metrics"
            },
            "drawdown": {
                "term": "Drawdown",
                "definition": "Calo massimo del valore del portafoglio dal picco precedente",
                "category": "risk"
            },
            "winrate": {
                "term": "Win Rate",
                "definition": "Percentuale di trade vincenti sul totale",
                "category": "metrics"
            },
            "profit_factor": {
                "term": "Profit Factor",
                "definition": "Rapporto tra profitti totali e perdite totali (obiettivo > 1.5)",
                "category": "metrics"
            },
            "long_position": {
                "term": "Long Position",
                "definition": "Posizione che beneficia dall'aumento del prezzo dell'asset",
                "category": "trading"
            },
            "short_position": {
                "term": "Short Position",
                "definition": "Posizione che beneficia dal calo del prezzo dell'asset",
                "category": "trading"
            }
        }
    
    def get_term(self, term_key: str) -> Optional[Dict]:
        return self.terms.get(term_key)
    
    def add_term(self, term_key: str, term: str, definition: str, category: str):
        self.terms[term_key] = {
            "term": term,
            "definition": definition,
            "category": category
        }
    
    def get_all_terms(self) -> Dict:
        return self.terms


# Inizializza vocabolario
vocabulary = FinancialVocabulary()


# ==================== ANALYZER ENGINE ====================

class PortfolioAnalyzer:
    """Motore di analisi del portafoglio"""
    
    def __init__(self, summary: Dict, positions: List[Dict]):
        self.summary = summary
        self.positions = positions
    
    def get_summary_text(self) -> str:
        """Genera testo riassuntivo del portafoglio"""
        if not self.summary:
            return "Dati non disponibili"
        
        total_value = self.summary.get("total_value", 0)
        total_pnl = self.summary.get("total_pnl", 0)
        total_return = self.summary.get("total_return_pct", 0)
        daily_pnl = self.summary.get("daily_pnl", 0)
        num_positions = self.summary.get("num_positions", 0)
        
        return f"""
**Portafoglio Totale:**
- Valore: {total_value:,.2f} USDT
- P&L Totale: {total_pnl:+,.2f} USDT ({total_return:+.2f}%)
- P&L Giornaliero: {daily_pnl:+,.2f} USDT
- Posizioni Aperte: {num_positions}
"""
    
    def get_positions_text(self) -> str:
        """Genera testo delle posizioni"""
        if not self.positions:
            return "Nessuna posizione"
        
        text = "\n**Posizioni Aperte:**\n"
        
        # Ordina per P&L
        sorted_positions = sorted(
            self.positions, 
            key=lambda x: x.get("unrealized_pnl", 0), 
            reverse=True
        )
        
        for pos in sorted_positions:
            symbol = pos.get("symbol", "")
            side = pos.get("side", "")
            pnl = pos.get("unrealized_pnl", 0)
            entry = pos.get("entry_price", 0)
            current = pos.get("current_price", 0)
            
            emoji = "🟢" if pnl > 0 else "🔴"
            text += f"- {emoji} {symbol} ({side}): {pnl:+,.2f} USDT (entry: {entry:.2f}, current: {current:.2f})\n"
        
        return text
    
    def get_worst_positions(self, limit: int = 3) -> List[Dict]:
        """Restituisce le peggiori posizioni"""
        if not self.positions:
            return []
        return sorted(self.positions, key=lambda x: x.get("unrealized_pnl", 0))[:limit]
    
    def get_best_positions(self, limit: int = 3) -> List[Dict]:
        """Restituisce le migliori posizioni"""
        if not self.positions:
            return []
        return sorted(self.positions, key=lambda x: x.get("unrealized_pnl", 0), reverse=True)[:limit]
    
    def get_risk_analysis(self) -> str:
        """Analisi dei rischi"""
        if not self.positions:
            return "Nessuna posizione da analizzare"
        
        long_count = sum(1 for p in self.positions if p.get("side") == "LONG")
        no_stop_loss = sum(1 for p in self.positions if not p.get("stop_loss"))
        
        text = f"""
**Analisi Rischio:**
- Posizioni LONG: {long_count}/{len(self.positions)}
- Posizioni senza stop-loss: {no_stop_loss}/{len(self.positions)}
"""
        
        if no_stop_loss > len(self.positions) * 0.7:
            text += "\n⚠️ **ATTENZIONE**: La maggior parte delle posizioni non ha stop-loss attivo!\n"
        
        return text


class NewsAnalyzer:
    """Motore di analisi delle notizie"""
    
    def __init__(self, news_data: Dict):
        self.news = news_data.get("news", [])
    
    def get_summary_text(self) -> str:
        """Genera testo riassuntivo delle notizie"""
        if not self.news:
            return "Nessuna notizia disponibile"
        
        positive = [n for n in self.news if n.get("sentiment") == "positive"]
        negative = [n for n in self.news if n.get("sentiment") == "negative"]
        neutral = [n for n in self.news if n.get("sentiment") == "neutral"]
        
        avg_sentiment = sum(n.get("sentiment_score", 0) for n in self.news) / len(self.news)
        
        text = f"""
**Notizie Totali:** {len(self.news)}
- 🟢 Positive: {len(positive)}
- 🔴 Negative: {len(negative)}
- ➖ Neutrali: {len(neutral)}

**Sentiment Medio:** {avg_sentiment:.2f} ({'Positivo' if avg_sentiment > 0.3 else 'Negativo' if avg_sentiment < -0.3 else 'Neutro'})
"""
        
        text += "\n**Notizie Recenti:**\n"
        for news in self.news[:5]:
            emoji = "🟢" if news.get("sentiment") == "positive" else "🔴" if news.get("sentiment") == "negative" else "➖"
            text += f"- {emoji} {news.get('title', '')[:60]}...\n"
        
        return text
    
    def get_by_symbol(self, symbol: str) -> List[Dict]:
        """Filtra notizie per simbolo"""
        return [
            n for n in self.news 
            if symbol.upper() in [s.replace("/USDT", "").replace("USDT/", "") for s in n.get("symbols", [])]
        ]


# ==================== AI RESPONSE ENGINE ====================

def generate_analysis(
    portfolio: PortfolioAnalyzer, 
    news: NewsAnalyzer, 
    question: str,
    use_openai: bool = True
) -> str:
    """Genera analisi basata su domanda"""
    
    # Prepara dati
    portfolio_text = portfolio.get_summary_text() + "\n" + portfolio.get_positions_text()
    news_text = news.get_summary_text()
    risk_text = portfolio.get_risk_analysis()
    
    # Se OpenAI non disponibile, usa analisi locale
    if not use_openai:
        return generate_local_analysis(portfolio, news, question)
    
    # Prepara prompt per GPT
    prompt = f"""
Sei un assistente finanziario AI esperto in crypto trading.
Analizza il portafoglio e le notizie, poi rispondi alla domanda.

## PORTAFOGLIO:
{portfolio_text}

## ANALISI RISCHIO:
{risk_text}

## NOTIZIE E SENTIMENT:
{news_text}

## DOMANDA:
{question}

Istruzioni:
1. Rispondi in italiano in modo chiaro e professionale
2. Evidenzia i numeri chiave
3. Suggerisci azioni senza decidere al posto dell'utente
4. Mantieni il linguaggio semplice ma preciso
"""
    
    # Prova a usare OpenAI
    try:
        import os
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Sei un assistente finanziario esperto in crypto."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.5
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # Fallback a analisi locale
        return generate_local_analysis(portfolio, news, question)


def generate_local_analysis(
    portfolio: PortfolioAnalyzer, 
    news: NewsAnalyzer, 
    question: str
) -> str:
    """Genera analisi locale senza OpenAI"""
    
    question_lower = question.lower()
    response = []
    
    # Domande generiche sullo stato
    if any(word in question_lower for word in ["stato", "come va", "overview", "riepilogo"]):
        response.append("## 📊 Stato del Portafoglio\n")
        response.append(portfolio.get_summary_text())
        
        # Evidenzia best/worst
        best = portfolio.get_best_positions(1)
        worst = portfolio.get_worst_positions(1)
        
        if best:
            response.append(f"\n✅ **Miglior performer**: {best[0].get('symbol')} con P&L di {best[0].get('unrealized_pnl'):+,.2f} USDT")
        if worst:
            response.append(f"\n⚠️ **Peggiore**: {worst[0].get('symbol')} con P&L di {worst[0].get('unrealized_pnl'):+,.2f} USDT")
    
    # Domande su rischi
    elif any(word in question_lower for word in ["rischio", "pericolo", "stop loss", "drawdown"]):
        response.append("## ⚠️ Analisi Rischio\n")
        response.append(portfolio.get_risk_analysis())
        
        # Consigli rischio
        worst = portfolio.get_worst_positions(3)
        if worst:
            response.append("\n**Posizioni a rischio da monitorare:**\n")
            for pos in worst:
                pnl = pos.get("unrealized_pnl", 0)
                symbol = pos.get("symbol", "")
                if pnl < -5000:
                    response.append(f"- 🔴 {symbol}: {pnl:+,.2f} USDT - Considera stop-loss\n")
    
    # Domande su notizie
    elif any(word in question_lower for word in ["notizia", "sentiment", "news", "mercato"]):
        response.append("## 📰 Analisi Notizie\n")
        response.append(news.get_summary_text())
    
    # Domande su specifico asset
    else:
        # Estrai simbolo dalla domanda
        symbols = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "MATIC", "ATOM", "DOT", "UNI"]
        found_symbol = None
        for symbol in symbols:
            if symbol.lower() in question_lower:
                found_symbol = symbol
                break
        
        if found_symbol:
            response.append(f"## 📈 Analisi {found_symbol}USDT\n")
            
            # Trova posizione
            for pos in portfolio.positions:
                if found_symbol in pos.get("symbol", ""):
                    pnl = pos.get("unrealized_pnl", 0)
                    entry = pos.get("entry_price", 0)
                    current = pos.get("current_price", 0)
                    pnl_pct = ((current - entry) / entry) * 100
                    
                    status = "🟢" if pnl > 0 else "🔴"
                    response.append(f"{status} **P&L**: {pnl:+,.2f} USDT ({pnl_pct:+.2f}%)\n")
                    response.append(f"- Entry: {entry:.2f}\n")
                    response.append(f"- Current: {current:.2f}\n")
                    
                    if pnl < 0:
                        response.append("\n⚠️ **Suggerimento**: Monitora da vicino questa posizione\n")
                    break
            
            # Cerca notizie
            asset_news = news.get_by_symbol(found_symbol)
            if asset_news:
                response.append(f"\n**Notizie su {found_symbol}:**\n")
                for n in asset_news[:3]:
                    emoji = "🟢" if n.get("sentiment") == "positive" else "🔴"
                    response.append(f"{emoji} {n.get('title', '')[:50]}...\n")
        else:
            response.append("## 📊 Analisi Completa\n")
            response.append(portfolio.get_summary_text())
            response.append("\n")
            response.append(news.get_summary_text())
    
    # Aggiungi vocabolario se rilevante
    response.append("\n---\n")
    response.append("*Vocabolario aggiornato: usa il sidebar per consultare i termini finanziari.*")
    
    return "".join(response)


# ==================== STREAMLIT UI ====================

def main():
    """Interfaccia principale Streamlit"""
    
    # CSS personalizzato
    st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    .metric-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive { color: #22c55e; }
    .negative { color: #ef4444; }
    .warning { color: #f59e0b; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🤖 AI Financial Assistant")
    st.markdown("*Assistente finanziario AI per analisi del portafoglio crypto*")
    st.markdown("---")
    
    # Sidebar - Vocabolario
    with st.sidebar:
        st.header("📚 Vocabolario Finanziario")
        
        search_term = st.text_input("Cerca termine", "")
        
        if search_term:
            terms = vocabulary.get_all_terms()
            filtered = {k: v for k, v in terms.items() if search_term.lower() in v["term"].lower() or search_term.lower() in v["definition"].lower()}
        else:
            filtered = vocabulary.get_all_terms()
        
        for key, value in filtered.items():
            with st.expander(f"{value['term']}"):
                st.caption(f"**Categoria**: {value['category']}")
                st.write(value['definition'])
        
        st.markdown("---")
        st.caption("💡 *Termini finanziari aggiornati automaticamente*")
    
    # Recupera dati
    with st.spinner("Caricamento dati..."):
        summary = fetch_portfolio_summary()
        positions = fetch_positions()
        news_data = fetch_news()
    
    # Crea analizzatori
    portfolio = PortfolioAnalyzer(summary, positions)
    news = NewsAnalyzer(news_data)
    
    # Layout a colonne
    col1, col2, col3, col4 = st.columns(4)
    
    if summary:
        with col1:
            st.metric(
                "Valore Totale", 
                f"{summary.get('total_value', 0):,.0f} USDT"
            )
        with col2:
            pnl = summary.get('total_pnl', 0)
            st.metric(
                "P&L Totale", 
                f"{pnl:+,.0f} USDT",
                delta=f"{summary.get('total_return_pct', 0):+.1f}%"
            )
        with col3:
            st.metric(
                "P&L Giornaliero", 
                f"{summary.get('daily_pnl', 0):+,.2f} USDT",
                delta=f"{summary.get('daily_return_pct', 0)*100:+.2f}%"
            )
        with col4:
            st.metric(
                "Posizioni", 
                f"{summary.get('num_positions', 0)}"
            )
    
    st.markdown("---")
    
    # Tabs per contenuti
    tab1, tab2, tab3 = st.tabs(["💬 Chat con AI", "📊 Dettagli Portfolio", "📰 Notizie"])
    
    with tab1:
        st.subheader("Fai una domanda al tuo assistente")
        
        # Domande suggerite
        st.markdown("**Domande suggerite:**")
        suggestions = [
            "Come va il portafoglio oggi?",
            "Quali sono i rischi principali?",
            "Come sta performando BTC?",
            "Qual è il sentiment sul mercato?"
        ]
        
        cols = st.columns(2)
        for i, sug in enumerate(suggestions):
            if cols[i % 2].button(sug):
                st.session_state['question'] = sug
        
        # Input domanda
        question = st.text_input(
            "Scrivi la tua domanda:", 
            value=st.session_state.get('question', ''),
            key="question_input"
        )
        
        if st.button("Analizza", type="primary"):
            if question:
                with st.spinner("Analisi in corso..."):
                    response = generate_analysis(portfolio, news, question, use_openai=False)
                    
                    st.markdown("### 💡 Risposta")
                    st.markdown(response)
            else:
                st.warning("Scrivi una domanda per iniziare!")
    
    with tab2:
        st.subheader("📊 Dettagli Posizioni")
        
        if positions:
            # Tabella posizioni
            import pandas as pd
            
            df = pd.DataFrame(positions)
            df['pnl_emoji'] = df['unrealized_pnl'].apply(lambda x: "🟢" if x > 0 else "🔴")
            
            # Formatta colonne
            display_df = df[['pnl_emoji', 'symbol', 'side', 'quantity', 'entry_price', 
                           'current_price', 'unrealized_pnl']].copy()
            display_df.columns = ['📍', 'Symbol', 'Side', 'Qty', 'Entry', 'Current', 'P&L (USDT)']
            
            st.dataframe(
                display_df.style.format({
                    'Entry': '{:.2f}',
                    'Current': '{:.2f}', 
                    'P&L (USDT)': '{:+,.2f}'
                }),
                use_container_width=True
            )
            
            # Grafico P&L
            st.subheader("📈 P&L per Posizione")
            
            chart_data = df[['symbol', 'unrealized_pnl']].sort_values('unrealized_pnl')
            st.bar_chart(chart_data.set_index('symbol'))
        else:
            st.info("Nessuna posizione trovata")
    
    with tab3:
        st.subheader("📰 Notizie e Sentiment")
        
        if news_data.get("news"):
            for item in news_data["news"]:
                sentiment = item.get("sentiment", "neutral")
                emoji = "🟢" if sentiment == "positive" else "🔴" if sentiment == "negative" else "➖"
                
                with st.expander(f"{emoji} {item.get('title', 'Titolo non disponibile')}"):
                    st.markdown(f"**Fonte:** {item.get('source', 'N/A')}")
                    st.markdown(f"**Sentiment:** {item.get('sentiment_score', 0):.2f}")
                    st.markdown(f"**Simboli:** {', '.join(item.get('symbols', []))}")
                    st.markdown(f"**Categoria:** {item.get('category', 'N/A')}")
                    st.markdown(f"\n_{item.get('summary', '')}_")
        else:
            st.info("Nessuna notizia disponibile")
    
    # Footer
    st.markdown("---")
    st.caption(f"🕐 Ultimo aggiornamento: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


if __name__ == "__main__":
    main()
