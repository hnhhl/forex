import sys
sys.path.append('src')
from src.core.ultimate_xau_system import UltimateXAUSystem, SystemConfig

config = SystemConfig()
config.symbol = 'XAUUSDc'
system = UltimateXAUSystem(config)

print('Testing AI3.0 with Permanent Hybrid Logic:')
print('='*50)

signals = []
for i in range(5):
    signal = system.generate_signal('XAUUSDc')
    signals.append(signal)
    
    action = signal.get('action')
    confidence = signal.get('confidence', 0)
    method = signal.get('ensemble_method', 'unknown')
    
    # Get hybrid metrics
    metrics = signal.get('hybrid_metrics', {})
    consensus = metrics.get('hybrid_consensus', 0)
    
    print(f'Signal {i+1}: {action} ({confidence:.1%}) | Consensus: {consensus:.1%}')

print('='*50)

# Summary
avg_confidence = sum(s.get('confidence', 0) for s in signals) / len(signals)
actions = [s.get('action') for s in signals]
unique_actions = set(actions)
methods = [s.get('ensemble_method') for s in signals]

print(f'Average Confidence: {avg_confidence:.1%}')
print(f'Signal Types: {unique_actions}')
print(f'Method Used: {methods[0]}')

if 'hybrid_ai2_ai3_consensus' in methods:
    print('✅ HYBRID LOGIC PERMANENTLY APPLIED!')
    print('✅ AI3.0 now uses AI2.0 weighted + AI3.0 consensus!')
else:
    print('❌ Hybrid logic not detected')

print('='*50) 