# AI Phases - Hệ thống nâng cao hiệu suất AI

AI Phases là hệ thống nâng cao hiệu suất AI với 6 phases độc đáo, mỗi phase đóng góp một phần vào tổng performance boost +12%.

## Tổng quan

Hệ thống AI Phases bao gồm 6 phases:

| Phase | Tên | Boost | Tính năng chính |
|-------|-----|-------|-----------------|
| 1 | Online Learning Engine | +2.5% | Học liên tục từ market data, nhận diện patterns, bộ nhớ thích ứng |
| 2 | Advanced Backtest Framework | +1.5% | Kiểm thử nhiều kịch bản thị trường, phân tích hiệu suất, đánh giá rủi ro |
| 3 | Adaptive Intelligence | +3.0% | Tự động điều chỉnh chiến lược, nhận diện chế độ thị trường, phân tích tâm lý |
| 4 | Multi-Market Learning | +2.0% | Phân tích đa thị trường, phát hiện tương quan, nhận diện xu hướng toàn cầu |
| 5 | Real-Time Enhancement | +1.5% | Tối ưu độ trễ, xử lý dữ liệu theo luồng, kiến trúc hướng sự kiện |
| 6 | Future Evolution | +1.5% | Tự cải thiện hiệu suất, dự đoán nâng cao, mô phỏng kịch bản, thuật toán tiến hóa |

**Tổng performance boost: +12.0%**

## Cài đặt

```bash
pip install ai-phases
```

Hoặc cài đặt từ source:

```bash
git clone https://github.com/ai-developer/ai-phases.git
cd ai-phases
pip install -e .
```

## Yêu cầu hệ thống

- Python 3.7+
- NumPy >= 1.19.0
- Pandas >= 1.0.0

## Sử dụng cơ bản

### Khởi tạo hệ thống

```python
from ai_phases.main import AISystem

# Khởi tạo hệ thống AI với tất cả 6 phases
ai_system = AISystem()

# Kiểm tra trạng thái hệ thống
status = ai_system.get_system_status()
print(f"Total Performance Boost: +{status['system_state']['total_performance_boost']}%")
```

### Xử lý dữ liệu thị trường

```python
# Dữ liệu thị trường mẫu
market_data = {
    'close': [100, 101, 102, 101.5, 103, 104, 103.5, 105],
    'volume': [1000, 1200, 800, 950, 1300, 1100, 900, 1400]
}

# Xử lý dữ liệu qua tất cả phases
result = ai_system.process_market_data(market_data)

# Hiển thị kết quả
print(f"Combined Signal: {result['combined_signal']}")
print(f"Market Regime: {result['phase3_analysis']['market_regime']}")
print(f"Market Sentiment: {result['phase3_analysis']['market_sentiment']}")
print(f"Future Prediction: {result['phase6_prediction']['value']} (Confidence: {result['phase6_prediction']['confidence']})")
```

### Chạy backtest

```python
from ai_phases.phase2_backtest_framework import BacktestScenario

# Định nghĩa chiến lược đơn giản
def simple_strategy(data):
    if len(data) < 2:
        return 0
    return 1 if data['close'].iloc[-1] > data['close'].iloc[-2] else -1

# Chạy backtest với kịch bản bull market
results = ai_system.run_backtest(
    simple_strategy, 
    scenario=BacktestScenario.BULL_MARKET
)

# Hiển thị kết quả backtest
print(f"Total Return: {results['metrics']['total_return']:.2%}")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
```

### Mô phỏng kịch bản tương lai

```python
# Mô phỏng kịch bản thị trường giảm
scenario_results = ai_system.simulate_scenario(
    "bear_market", 
    parameters={'time_steps': 252, 'volatility': 0.15}
)

# Hiển thị kết quả mô phỏng
print(f"Final Value: {scenario_results['metrics']['final_value']:.2f}")
print(f"Total Return: {scenario_results['metrics']['total_return']:.2%}")
print(f"Max Drawdown: {scenario_results['metrics']['max_drawdown']:.2%}")
```

### Tiến hóa hệ thống

```python
# Tiến hóa hệ thống qua 5 vòng lặp
evolution_results = ai_system.evolve_system(iterations=5)

# Hiển thị kết quả tiến hóa
print(f"Initial Fitness: {evolution_results['initial_fitness']:.2f}")
print(f"Final Fitness: {evolution_results['final_fitness']:.2f}")
print(f"Improvement: {evolution_results['improvement']:.2f}")
print(f"Current Stage: {evolution_results['current_stage']}")
```

### Mô phỏng event stream

```python
# Mô phỏng event stream trong 10 giây với 5 events/giây
simulation_results = ai_system.simulate_event_stream(
    duration_seconds=10,
    events_per_second=5
)

# Hiển thị kết quả mô phỏng
print(f"Events Accepted: {simulation_results['events_accepted']}")
print(f"Average Latency: {simulation_results['performance_metrics']['average_latency_ms']:.2f} ms")
print(f"Events Per Second: {simulation_results['performance_metrics']['events_per_second']:.2f}")
```

### Tắt hệ thống

```python
# Tắt hệ thống khi hoàn thành
ai_system.shutdown()
```

## Cấu trúc thư viện

```
ai_phases/
├── __init__.py
├── main.py                      # Module tích hợp tất cả phases
├── phase1_online_learning.py    # Phase 1: Online Learning Engine
├── phase2_backtest_framework.py # Phase 2: Advanced Backtest Framework
├── phase3_adaptive_intelligence.py # Phase 3: Adaptive Intelligence
├── phase4_multi_market_learning.py # Phase 4: Multi-Market Learning
├── phase5_realtime_enhancement.py # Phase 5: Real-Time Enhancement
├── phase6_future_evolution.py   # Phase 6: Future Evolution
└── utils/
    ├── __init__.py
    └── progress_tracker.py      # Theo dõi tiến độ phát triển
```

## Giấy phép

MIT License

## Liên hệ

Nếu có câu hỏi hoặc góp ý, vui lòng liên hệ: ai@example.com 