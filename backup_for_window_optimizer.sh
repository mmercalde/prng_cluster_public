#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Creating in-place backups with timestamp: ${TIMESTAMP}"
echo ""

# Backup coordinator.py
if [ -f "coordinator.py" ]; then
    cp coordinator.py "coordinator.py.bak_window_opt_${TIMESTAMP}"
    echo "✅ Backed up: coordinator.py → coordinator.py.bak_window_opt_${TIMESTAMP}"
fi

# Backup sieve_filter.py
if [ -f "sieve_filter.py" ]; then
    cp sieve_filter.py "sieve_filter.py.bak_window_opt_${TIMESTAMP}"
    echo "✅ Backed up: sieve_filter.py → sieve_filter.py.bak_window_opt_${TIMESTAMP}"
fi

# Backup prng_registry.py
if [ -f "prng_registry.py" ]; then
    cp prng_registry.py "prng_registry.py.bak_window_opt_${TIMESTAMP}"
    echo "✅ Backed up: prng_registry.py → prng_registry.py.bak_window_opt_${TIMESTAMP}"
fi

# Backup existing window optimizer files if they exist
if [ -f "window_optimizer.py" ]; then
    cp window_optimizer.py "window_optimizer.py.bak_${TIMESTAMP}"
    echo "✅ Backed up: window_optimizer.py → window_optimizer.py.bak_${TIMESTAMP}"
fi

if [ -f "window_optimizer_integration.py" ]; then
    cp window_optimizer_integration.py "window_optimizer_integration.py.bak_${TIMESTAMP}"
    echo "✅ Backed up: window_optimizer_integration.py → window_optimizer_integration.py.bak_${TIMESTAMP}"
fi

echo ""
echo "✅ All backups complete!"
echo ""
echo "To restore coordinator.py:"
echo "  cp coordinator.py.bak_window_opt_${TIMESTAMP} coordinator.py"
echo ""
echo "To restore sieve_filter.py:"
echo "  cp sieve_filter.py.bak_window_opt_${TIMESTAMP} sieve_filter.py"
echo ""
echo "To restore prng_registry.py:"
echo "  cp prng_registry.py.bak_window_opt_${TIMESTAMP} prng_registry.py"
echo ""

