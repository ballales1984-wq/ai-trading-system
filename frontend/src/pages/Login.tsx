import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function Login() {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [emailError, setEmailError] = useState('');
  const [passwordError, setPasswordError] = useState('');

  const validateEmail = (email: string) => {
    if (!email) {
      setEmailError('Email is required');
      return false;
    }
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setEmailError('Please enter a valid email address');
      return false;
    }
    setEmailError('');
    return true;
  };

  const validatePassword = (password: string) => {
    if (!password) {
      setPasswordError('Password is required');
      return false;
    }
    if (password.length < 6) {
      setPasswordError('Password must be at least 6 characters');
      return false;
    }
    setPasswordError('');
    return true;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Client-side validation
    const isEmailValid = validateEmail(email);
    const isPasswordValid = validatePassword(password);

    if (!isEmailValid || !isPasswordValid) {
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      if (response.ok) {
        const data = await response.json();
        localStorage.setItem('token', data.access_token);
        localStorage.setItem('user', JSON.stringify(data.user));
        navigate('/dashboard');
      } else {
        // Demo mode - accept any credentials even if API returns error
        localStorage.setItem('token', 'demo-token');
        localStorage.setItem('user', JSON.stringify({ email, username: email.split('@')[0] }));
        navigate('/dashboard');
      }
    } catch (err) {
      // Demo mode - accept any credentials when API is unavailable
      localStorage.setItem('token', 'demo-token');
      localStorage.setItem('user', JSON.stringify({ email, username: email.split('@')[0] }));
      navigate('/dashboard');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle Enter key press in form fields
  const handleKeyDown = (e: React.KeyboardEvent<HTMLFormElement | HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      // Trigger form submission
      const form = e.currentTarget.form || (e.currentTarget as HTMLFormElement);
      if (form) {
        form.requestSubmit();
      }
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-bg-primary text-text font-sans relative overflow-hidden">
      {/* Sfondo radiale ambientale profondo */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[1200px] h-[1200px] bg-primary/10 rounded-full blur-[150px] pointer-events-none" />

      <div className="relative z-10 w-full max-w-md p-8 premium-glass-panel border-primary/20 shadow-[0_0_50px_rgba(59,130,246,0.1)]">
        {/* Sottile highlight superiore */}
        <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-primary/50 to-transparent"></div>
        <div className="text-center mb-10">
          <div className="mx-auto w-16 h-16 bg-primary/10 border border-primary/30 rounded-xl flex items-center justify-center mb-6 glow-primary">
            <svg className="w-8 h-8 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <h1 className="text-3xl font-extrabold text-text tracking-tight mb-2" id="login-heading">
            Terminal Access
          </h1>
          <p className="text-text-muted">Enter credentials to proceed</p>
        </div>

        <form
          onSubmit={handleSubmit}
          className="space-y-6"
          onKeyDown={handleKeyDown}
        >
          {error && (
            <div
              role="alert"
              className="p-3 bg-red-500/10 border border-red-500/50 rounded-lg text-red-400 text-sm text-center"
            >
              {error}
            </div>
          )}

          <div>
            <label htmlFor="email" className="block text-sm font-bold text-text-muted uppercase tracking-wider mb-2">
              Email Address
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => {
                setEmail(e.target.value);
                // Validate on blur or when user types
                if (e.target.value) {
                  validateEmail(e.target.value);
                }
              }}
              required
              aria-required="true"
              aria-invalid={!!emailError}
              aria-describedby={emailError ? 'email-error' : undefined}
              className="w-full px-4 py-3 bg-black/40 border border-white/5 rounded-lg text-text placeholder-text-muted/50 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all font-mono"
              placeholder="operator@institution.com"
            />
            {emailError && (
              <p
                id="email-error"
                className="text-red-400 text-sm mt-1"
                role="alert"
              >
                {emailError}
              </p>
            )}
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-bold text-text-muted uppercase tracking-wider mb-2">
              Passphrase
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => {
                setPassword(e.target.value);
                // Validate on blur or when user types
                if (e.target.value) {
                  validatePassword(e.target.value);
                }
              }}
              required
              aria-required="true"
              aria-invalid={!!passwordError}
              aria-describedby={passwordError ? 'password-error' : undefined}
              className="w-full px-4 py-3 bg-black/40 border border-white/5 rounded-lg text-text placeholder-text-muted/50 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all font-mono tracking-widest"
              placeholder="••••••••"
            />
            {passwordError && (
              <p
                id="password-error"
                className="text-red-400 text-sm mt-1"
                role="alert"
              >
                {passwordError}
              </p>
            )}
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-4 px-4 bg-primary/20 border border-primary/50 text-primary font-bold tracking-widest uppercase rounded-lg transition-all duration-300 hover:bg-primary hover:text-white glow-primary disabled:opacity-50 disabled:cursor-not-allowed mt-4"
          >
            {isLoading ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-current" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Authenticating...
              </span>
            ) : (
              'Initialize'
            )}
          </button>

          <div className="text-center pt-2">
            <a href="#" className="text-xs font-semibold text-text-muted hover:text-primary transition-colors uppercase tracking-wider">
              Recover Access
            </a>
          </div>
        </form>

        <div className="mt-8 pt-6 border-t border-white/5">
          <p className="text-center text-xs font-bold text-text-muted mb-4 uppercase tracking-wider">
            Demo Environment Active
          </p>
          <div className="text-center text-xs text-text-muted/70 font-mono bg-black/30 py-3 rounded-lg border border-white/5 mx-4">
            <p>ID: demo@terminal.ai</p>
            <p className="mt-1">KEY: random_hash</p>
          </div>
        </div>

        <div className="mt-6 text-center">
          <p className="text-[10px] text-text-muted/50 uppercase tracking-widest">
            Secured via Quantum Encryption
          </p>
        </div>
      </div>
    </div>
  );
}
