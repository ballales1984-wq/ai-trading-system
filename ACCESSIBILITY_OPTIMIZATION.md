# ♿ Accessibility Optimization Guide
> Risolvere problemi di accessibilità per AI Trading System

## 📊 Problemi Identificati

### ⚠️ Critici
- **Color Contrast**: 839ms - Elementi non rispettano WCAG AA
- **Focus Management**: Elementi nascosti non gestiti correttamente
- **ARIA Attributes**: Label e description mancanti
- **Keyboard Navigation**: Tab order non logico

## 🎯 Soluzioni Immediate

### 1. Color Contrast Optimization
```css
/* PROBLEMA: Contrasto insufficiente */
.card-header {
  background: #f0f0f0;
  color: #ffffff;
}

/* SOLUZIONE: Contrasto WCAG AA (4.5:1) */
.card-header {
  background: #1a1a1a;
  color: #ffffff;
}

/* Oppure usa variabili CSS personalizzate */
:root {
  --text-primary: #ffffff;
  --bg-primary: #1a1a1a;
  --text-secondary: #666666;
  --bg-secondary: #f8f9fa;
}
```

### 2. Focus Management
```typescript
// PROBLEMA: Elementi nascosti non focusabili
const HiddenElement = () => (
  <div aria-hidden="true">
    Contenuto nascosto
  </div>
);

// SOLUZIONE: Gestione corretta del focus
const HiddenElement = () => (
  <div 
    aria-hidden="true"
    tabIndex={-1} // Non focusabile
    aria-describedby="hidden-content-description"
  >
    <span id="hidden-content-description" className="sr-only">
      Questo contenuto è nascosto agli utenti visivi
    </span>
    Contenuto nascosto
  </div>
);
```

### 3. ARIA Attributes
```typescript
// PROBLEMA: Mancanza di label e description
<button onClick={handleAction}>
  <Icon />
  Azione
</button>

// SOLUZIONE: Accessibilità completa
<button 
  onClick={handleAction}
  aria-label="Esegui trade su BTC/USDT"
  aria-describedby="trade-description"
>
  <Icon />
  Azione
</button>
<div id="trade-description" className="sr-only">
  Esegue un ordine di trading automatico sulla coppia BTC/USDT con le impostazioni correnti
</div>
```

### 4. Keyboard Navigation
```typescript
// PROBLEMA: Tab order non logico
<div>
  <button>Salva</button>
  <input type="text" />
  <button>Annulla</button>
</div>

// SOLUZIONE: Tab order corretto
<div>
  <button>Salva</button>
  <input type="text" />
  <button>Annulla</button>
</div>

// Con tabindex espliciti
<div>
  <button tabIndex={1}>Salva</button>
  <input type="text" tabIndex={2} />
  <button tabIndex={3}>Annulla</button>
</div>
```

### 5. Screen Reader Support
```typescript
// PROBLEMA: Content non accessibile
const PortfolioValue = () => (
  <div>
    <span className="text-green-500">+€1,234.56</span>
  </div>
);

// SOLUZIONE: Accessibilità completa
const PortfolioValue = () => (
  <div aria-live="polite" aria-atomic="true">
    <span className="text-green-500">+€1,234.56</span>
    <span className="sr-only">
      Valore portfolio: positivo mille duecento trentaquattro virgola cinquantasei
    </span>
  </div>
);
```

### 6. Forms Accessibility
```typescript
// PROBLEMA: Form non accessibile
<input 
  type="number"
  placeholder="Quantità"
  onChange={handleChange}
/>

// SOLUZIONE: Form accessibile completo
<input 
  type="number"
  placeholder="Quantità"
  onChange={handleChange}
  aria-label="Quantità da tradare"
  aria-describedby="quantity-help"
  required
  min={0.001}
  step={0.001}
/>
<div id="quantity-help" className="text-sm text-gray-500">
  Inserisci la quantità in BTC, minimo 0.001
</div>
```

## 🎯 Metriche Target WCAG 2.1 AA

### Obiettivi di Accessibilità:
- **Color Contrast**: 4.5:1 ratio minimo
- **Keyboard Accessible**: Tutte le funzioni accessibili da tastiera
- **Screen Reader**: Content descritto correttamente
- **Focus Management**: Focus logico e visibile
- **Forms**: Label e help text completi

### 📊 Testing Tools

### 1. Browser Extensions
- **axe DevTools**: Analisi automatica accessibilità
- **WAVE**: Web Accessibility Evaluation Tool
- **Colour Contrast Analyser**: Verifica contrasti colori

### 2. Automated Testing
```bash
# Installa axe-core
npm install --save-dev @axe-core/core

# In unit test
import { axe, toHaveNoViolations } from '@axe-core/core';

test('portfolio accessibility', async () => {
  const results = await axe(document.body);
  expect(results).toHaveNoViolations();
});
```

### 3. Manual Testing Checklist
```markdown
- [ ] Test navigazione da tastiera (Tab, Shift+Tab, Enter, Space, Escape)
- [ ] Test screen reader (NVDA, JAWS, VoiceOver)
- [ ] Test zoom browser (200%, 400%)
- [ ] Test contrasto colori (Chrome DevTools)
- [ ] Test focus management (Tab order visibile)
- [ ] Test forms con validation errors
```

## 🚀 Piano di Azione

### Fase 1: Immediata (Oggi)
1. **Implementare contrasti colori WCAG AA**
2. **Aggiungere ARIA labels** a tutti i bottoni e input
3. **Correggere tab order** nei form
4. **Aggiungere screen reader support** con aria-live

### Fase 2: Medio Termine (Q1)
1. **Implementare focus trap** per modali e dialoghi
2. **Aggiungere skip links** per navigazione rapida
3. **Testare con screen reader** reali
4. **Implementare high contrast mode**

---

**Obiettivo: 100% WCAG 2.1 AA compliance** ♿
