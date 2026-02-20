# 🚀 HARDENING PLAN - Step by Step

## 4A. LATENCY ENGINEERING

### Step 1: uvloop Integration ✅ DONE
- [x] Add uvloop to requirements.txt (later)
- [x] Create `src/core/performance/event_loop.py` - Custom event loop with uvloop
- [ ] Update main.py to use optimized event loop
- [ ] Update WebSocket stream to use uvloop

### Step 2: WebSocket Batch Processing
- [ ] Create `src/core/performance/ring_buffer.py` - Lock-free ring buffer
- [ ] Update `app/market_data/websocket_stream.py` - Batch message processing
- [ ] Add message batching with configurable window (1ms, 5ms, 10ms)

### Step 3: DB Write Batching
- [ ] Create `src/core/performance/db_batcher.py` - Write batching for TimescaleDB
- [ ] Update `src/database_async_repository.py` - Use batched writes
- [ ] Add flush methods with timing control

### Step 4: Async Logging ✅ DONE
- [x] Create `src/core/performance/async_logging.py` - Queue-based async logging
- [x] Update `app/core/logging_production.py` - Use async logging
- [x] Configure log levels for production

---

## 4B. PERFORMANCE PROFILING

### Step 5: Profiling Infrastructure ✅ DONE
- [x] Create `src/core/performance/metrics.py` - Custom timing decorators
- [x] Add latency tracking to key functions

### Step 6: Prometheus Metrics
- [ ] Create `src/core/performance/prometheus_metrics.py` - Prometheus exporters
- [ ] Add metrics for:
  - [ ] order_execution_latency_seconds
  - [ ] signal_generation_time_seconds
  - [ ] db_write_latency_seconds
  - [ ] risk_check_time_seconds
  - [ ] websocket_message_rate
- [ ] Update docker-compose to include Prometheus

---

## 4C. SCALING

### Step 7: Redis Pub/Sub Setup
- [ ] Create `src/core/performance/message_bus.py` - Redis pub/sub wrapper
- [ ] Define channels: signals, orders, risk, execution
- [ ] Add message serialization (JSON/msgpack)

### Step 8: Microservices Structure (Optional)
- [ ] Create service wrappers for:
  - [ ] Signal Service
  - [ ] Risk Service
  - [ ] Execution Service

---

## Implementation Order

```
1. uvloop + Event Loop Optimization
2. Async Logging (quick win)
3. Profiling Infrastructure
4. Prometheus Metrics
5. Redis Pub/Sub
6. Ring Buffer + Batch Processing
7. DB Write Batching
```

---

*Start with Step 1: uvloop Integration*

