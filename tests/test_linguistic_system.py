"""
Integration Test: Linguistic System & News Flow
==============================================
Verifica il corretto funzionamento della filiera:
News -> Sentiment -> Concetti -> Risposta IA (Italiano)
"""

import sys
import os
import asyncio
import logging
from datetime import datetime

# Aggiungi root al path
sys.path.append(os.getcwd())

# Fix encoding per Windows console (emojis)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sentiment_news import SentimentAnalyzer
from advanced_ai_assistant import FinancialAIAssistant
from sentiment_concept_bridge import create_bridge
from shared_vocabulary import SHARED_FINANCIAL_CONCEPTS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_linguistic")

async def test_news_flow():
    print("\n" + "="*50)
    print("TEST: FLUSSO NOTIZIE E SENTIMENT")
    print("="*50)
    
    analyzer = SentimentAnalyzer()
    # Usiamo 'MARKET' per i test
    news = analyzer.fetch_news(assets=["MARKET"])
    
    if not news:
        print("[-] Errore: Nessuna notizia trovata (modalità demo attiva?)")
        # In demo mode, fetch_news dovrebbe comunque restituire dati mock
        return False
        
    print(f"[+] Trovate {len(news)} notizie.")
    print(f"[+] Esempio Sentiment: {news[0].sentiment:.2f} ({news[0].title[:50]}...)")
    return True

async def test_ai_assistant_italian():
    print("\n" + "="*50)
    print("TEST: AI ASSISTANT E VOCABOLARIO ITALIANO")
    print("="*50)
    
    assistant = FinancialAIAssistant()
    
    # Test 1: Spiegazione termine dal vocabolario condiviso
    term = "sharpe_ratio"
    explanation = assistant.explain_term("sharpe ratio")
    print(f"[+] Termine: {term}")
    print(f"[+] Risposta IA: {explanation[:150]}...")
    
    if "Sharpe" not in explanation or "rischio" not in explanation.lower():
        print("[-] Errore: Spiegazione non coerente o non in italiano.")
        return False
        
    # Test 2: Domanda generica sul mercato
    response = assistant.get_ai_response("Come va il mercato?")
    print(f"\n[+] Domanda: 'Come va il mercato?'")
    print(f"[+] Risposta IA: {response[:150]}...")
    
    if "Sentiment" not in response:
        print("[-] Errore: La risposta non contiene l'analisi del sentiment.")
        return False
        
    return True

async def test_learning_loop():
    print("\n" + "="*50)
    print("TEST: CICLO DI APPRENDIMENTO")
    print("="*50)
    
    assistant = FinancialAIAssistant()
    initial_count = len(assistant.learned_concepts)
    
    # Mock news con termine sconosciuto (univoco per evitare conflitti di persistenza)
    mock_news = [
        {
            "title": "Protocollo Quantum-Ledger-Nova lancia mainnet oggi",
            "summary": "Quantum-Ledger-Nova è una nuova tecnologia blockchain.",
            "source": "TestDetector"
        }
    ]
    
    assistant.learn_from_news(mock_news)
    new_count = len(assistant.learned_concepts)
    
    print(f"[+] Concetti appresi iniziali: {initial_count}")
    print(f"[+] Concetti appresi finali: {new_count}")
    
    # Cerchiamo il termine appreso (hyperdrive perché le parole < 4 sono saltate e minuscolo)
    if new_count > initial_count:
        print("[+] Successo: L'assistente ha appreso nuovi termini.")
        return True
    else:
        print("[-] Errore: Nessun nuovo termine appreso.")
        return False

async def run_all_tests():
    success = True
    
    if not await test_news_flow(): success = False
    if not await test_ai_assistant_italian(): success = False
    if not await test_learning_loop(): success = False
    
    print("\n" + "="*50)
    if success:
        print("ESITO FINALE: TUTTI I TEST SUPERATI!")
    else:
        print("ESITO FINALE: ALCUNI TEST FALLITI.")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(run_all_tests())
