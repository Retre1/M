## Summary

<!-- 1-3 bullet points describing the change -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Refactoring
- [ ] Configuration change

## Risk assessment

- [ ] Affects live trading logic
- [ ] Affects risk management
- [ ] Affects order execution
- [ ] Changes configuration schema
- [ ] Modifies state persistence

## Testing

- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Backtested on historical data
- [ ] Shadow-traded in demo environment

## Checklist

- [ ] No hardcoded credentials or secrets
- [ ] No `weights_only=False` in torch.load
- [ ] Error handling for all MT5 API calls
- [ ] WAL write before state mutation
- [ ] Logging for all critical code paths
