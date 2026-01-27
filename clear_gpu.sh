#!/bin/bash

# Script to clear GPU memory and kill processes using GPUs

set -e

echo "=== Checking GPU Status ==="
nvidia-smi

echo ""
echo "=== Finding processes using GPU ==="
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sort -u | tr '\n' ' ')

if [ -z "$GPU_PIDS" ]; then
    echo "No GPU processes found."
else
    echo "Found GPU processes: $GPU_PIDS"
    
    echo ""
    echo "=== Killing GPU processes ==="
    for pid in $GPU_PIDS; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "Killing PID $pid"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
fi

echo ""
echo "=== Finding Python/ML processes ==="
PYTHON_PIDS=$(ps aux | grep -E "(python|torch|vllm|accelerate)" | grep -v grep | awk '{print $2}' | sort -u | tr '\n' ' ')

if [ -z "$PYTHON_PIDS" ]; then
    echo "No Python/ML processes found."
else
    echo "Found Python/ML processes: $PYTHON_PIDS"
    for pid in $PYTHON_PIDS; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "Killing PID $pid"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
fi

echo ""
echo "=== Waiting for processes to terminate ==="
sleep 2

echo ""
echo "=== Final GPU Status ==="
nvidia-smi

echo ""
echo "=== Remaining Python/ML processes ==="
ps aux | grep -E "(python|torch|vllm|accelerate)" | grep -v grep || echo "None found."

echo ""
echo "=== Clearing debug folders ==="
rm -rf /home/ubuntu/spider/debug_chunk_history/*
rm -rf /home/ubuntu/spider/debug_loss/*
rm -rf /home/ubuntu/spider/debug_tool_exec/*
rm -rf /home/ubuntu/spider/debug_traj_b4_training/*
rm -rf /home/ubuntu/spider/debug_vllm_calls/*
echo "Cleared: debug_chunk_history, debug_loss, debug_tool_exec, debug_traj_b4_training, debug_vllm_calls"

echo ""
echo "Done!"

