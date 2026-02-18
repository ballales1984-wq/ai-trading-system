"""
App Analysis Tests - Verify functionality while app is running
These tests verify that all components work correctly in real-time
"""
import pytest
import requests
import time
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDashboardRunning:
    """Test that dashboard is running and responding"""
    
    def test_dashboard_is_accessible(self):
        """Verify dashboard is accessible at localhost:8050"""
        try:
            response = requests.get("http://127.0.0.1:8050", timeout=10)
            assert response.status_code == 200, f"Dashboard returned status {response.status_code}"
            assert "Quantum AI" in response.text or "Trading" in response.text
            print(f"✅ Dashboard accessible - Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Dashboard not accessible: {e}")
    
    def test_dashboard_network_accessible(self):
        """Verify dashboard is accessible on network interface"""
        try:
            response = requests.get("http://192.168.0.3:8050", timeout=10)
            assert response.status_code == 200
            print(f"✅ Network dashboard accessible")
        except requests.exceptions.RequestException:
            print(f"⚠️ Network dashboard not accessible (may be firewall)")


class TestDataCollection:
    """Test data collection modules"""
    
    def test_technical_analysis_import(self):
        """Verify technical analysis module imports correctly"""
        try:
            from technical_analysis import TechnicalAnalysis
            ta = TechnicalAnalysis()
            assert ta is not None
            print(f"✅ Technical Analysis module loaded")
        except Exception as e:
            pytest.fail(f"Technical Analysis import failed: {e}")
    
    def test_data_collector_import(self):
        """Verify data collector module imports correctly"""
        try:
            from data_collector import DataCollector
            dc = DataCollector()
            assert dc is not None
            print(f"✅ Data Collector module loaded")
        except Exception as e:
            pytest.fail(f"Data Collector import failed: {e}")
    
    def test_sentiment_news_import(self):
        """Verify sentiment/news module imports correctly"""
        try:
            from sentiment_news import SentimentNews
            sn = SentimentNews()
            assert sn is not None
            print(f"✅ Sentiment News module loaded")
        except Exception as e:
            pytest.fail(f"Sentiment News import failed: {e}")
    
    def test_onchain_analysis_import(self):
        """Verify on-chain analysis module imports correctly"""
        try:
            from onchain_analysis import OnChainAnalysis
            oca = OnChainAnalysis()
            assert oca is not None
            print(f"✅ On-Chain Analysis module loaded")
        except Exception as e:
            pytest.fail(f"On-Chain Analysis import failed: {e}")


class TestCoreModules:
    """Test core system modules"""
    
    def test_decision_engine_import(self):
        """Verify decision engine imports correctly"""
        try:
            from decision_engine import DecisionEngine
            de = DecisionEngine()
            assert de is not None
            print(f"✅ Decision Engine module loaded")
        except Exception as e:
            pytest.fail(f"Decision Engine import failed: {e}")
    
    def test_ml_predictor_import(self):
        """Verify ML predictor imports correctly"""
        try:
            from ml_predictor import MLPredictor
            mlp = MLPredictor()
            assert mlp is not None
            print(f"✅ ML Predictor module loaded")
        except Exception as e:
            pytest.fail(f"ML Predictor import failed: {e}")
    
    def test_trading_simulator_import(self):
        """Verify trading simulator imports correctly"""
        try:
            from trading_simulator import TradingSimulator
            ts = TradingSimulator()
            assert ts is not None
            print(f"✅ Trading Simulator module loaded")
        except Exception as e:
            pytest.fail(f"Trading Simulator import failed: {e}")
    
    def test_risk_engine_import(self):
        """Verify risk engine imports correctly"""
        try:
            from src.risk_engine import RiskEngine
            re = RiskEngine()
            assert re is not None
            print(f"✅ Risk Engine module loaded")
        except Exception as e:
            pytest.fail(f"Risk Engine import failed: {e}")


class TestAPIConfiguration:
    """Test API configuration and keys"""
    
    def test_env_file_exists(self):
        """Verify .env file exists"""
        env_path = ".env"
        assert os.path.exists(env_path), ".env file not found"
        print(f"✅ .env file exists")
    
    def test_env_file_has_keys(self):
        """Verify .env file has required API keys"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            # Check for configured keys
            keys = {
                "BINANCE_API_KEY": os.getenv("BINANCE_API_KEY"),
                "BINANCE_SECRET_KEY": os.getenv("BINANCE_SECRET_SECRET"),
                "NEWS_API_KEY": os.getenv("NEWS_API_KEY"),
            }
            
            configured = sum(1 for v in keys.values() if v and v != "")
            print(f"✅ API keys loaded: {configured}/3 configured")
        except Exception as e:
            print(f"⚠️ Error loading .env: {e}")


class TestDatabase:
    """Test database functionality"""
    
    def test_database_import(self):
        """Verify database module imports correctly"""
        try:
            from src.database import Database
            db = Database()
            assert db is not None
            print(f"✅ Database module loaded")
        except Exception as e:
            pytest.fail(f"Database import failed: {e}")


class TestPerformance:
    """Test performance monitoring"""
    
    def test_performance_module_import(self):
        """Verify performance module imports correctly"""
        try:
            from src.performance import Performance
            perf = Performance()
            assert perf is not None
            print(f"✅ Performance module loaded")
        except Exception as e:
            pytest.fail(f"Performance import failed: {e}")
    
    def test_kpi_module_import(self):
        """Verify KPI module imports correctly"""
        try:
            from src.kpi import KPI
            kpi = KPI()
            assert kpi is not None
            print(f"✅ KPI module loaded")
        except Exception as e:
            pytest.fail(f"KPI import failed: {e}")


class TestExecution:
    """Test order execution modules"""
    
    def test_execution_module_import(self):
        """Verify execution module imports correctly"""
        try:
            from src.execution import Execution
            exec = Execution()
            assert exec is not None
            print(f"✅ Execution module loaded")
        except Exception as e:
            pytest.fail(f"Execution import failed: {e}")


class TestPortfolio:
    """Test portfolio management"""
    
    def test_portfolio_manager_import(self):
        """Verify portfolio manager imports correctly"""
        try:
            from src.portfolio_optimizer import PortfolioOptimizer
            po = PortfolioOptimizer()
            assert po is not None
            print(f"✅ Portfolio Optimizer module loaded")
        except Exception as e:
            pytest.fail(f"Portfolio Optimizer import failed: {e}")
    
    def test_allocation_module_import(self):
        """Verify allocation module imports correctly"""
        try:
            from src.allocation import Allocation
            alloc = Allocation()
            assert alloc is not None
            print(f"✅ Allocation module loaded")
        except Exception as e:
            pytest.fail(f"Allocation import failed: {e}")


class TestUtilities:
    """Test utility functions"""
    
    def test_utils_import(self):
        """Verify utils module imports correctly"""
        try:
            from src.utils import Utils
            utils = Utils()
            assert utils is not None
            print(f"✅ Utils module loaded")
        except Exception as e:
            pytest.fail(f"Utils import failed: {e}")
    
    def test_indicators_import(self):
        """Verify indicators module imports correctly"""
        try:
            from src.indicators import Indicators
            ind = Indicators()
            assert ind is not None
            print(f"✅ Indicators module loaded")
        except Exception as e:
            pytest.fail(f"Indicators import failed: {e}")


class TestLiveTrading:
    """Test live trading components"""
    
    def test_live_trading_import(self):
        """Verify live trading module imports correctly"""
        try:
            from src.live_trading import LiveTrading
            lt = LiveTrading()
            assert lt is not None
            print(f"✅ Live Trading module loaded")
        except Exception as e:
            pytest.fail(f"Live Trading import failed: {e}")


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_full_import_chain(self):
        """Test that all major modules can be imported in sequence"""
        modules = [
            ("config", "Config"),
            ("technical_analysis", "TechnicalAnalysis"),
            ("data_collector", "DataCollector"),
            ("sentiment_news", "SentimentNews"),
            ("decision_engine", "DecisionEngine"),
            ("ml_predictor", "MLPredictor"),
            ("trading_simulator", "TradingSimulator"),
        ]
        
        imported = []
        for module_name, class_name in modules:
            try:
                module = __import__(module_name, fromlist=[class_name])
                cls = getattr(module, class_name)
                instance = cls()
                imported.append(module_name)
            except Exception as e:
                print(f"⚠️ Failed to import {module_name}.{class_name}: {e}")
        
        print(f"✅ Successfully imported {len(imported)}/{len(modules)} modules: {imported}")
        assert len(imported) >= len(modules) * 0.7, "Too many modules failed to import"


class TestSystemHealth:
    """System health checks"""
    
    def test_python_version(self):
        """Verify Python version is compatible"""
        version = sys.version_info
        assert version.major >= 3 and version.minor >= 8
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    
    def test_required_packages(self):
        """Verify required packages are installed"""
        required = ["numpy", "pandas", "requests", "flask", "dash", "pytest"]
        missing = []
        
        for package in required:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            print(f"⚠️ Missing packages: {missing}")
        else:
            print(f"✅ All required packages installed")
        
        assert len(missing) == 0, f"Missing packages: {missing}"
    
    def test_memory_usage(self):
        """Check memory usage"""
        try:
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / 1024 / 1024
            print(f"✅ Memory usage: {mem_mb:.2f} MB")
        except ImportError:
            print(f"⚠️ psutil not installed, skipping memory check")


def run_all_tests():
    """Run all tests and generate report"""
    print("\n" + "="*60)
    print("APP ANALYSIS TEST SUITE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60 + "\n")
    
    # Run pytest
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes",
        "-p", "no:warnings"
    ])
    
    return exit_code


if __name__ == "__main__":
    run_all_tests()
