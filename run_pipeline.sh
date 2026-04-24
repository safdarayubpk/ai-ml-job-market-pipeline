#!/bin/bash
# run_pipeline.sh — Trigger the pipeline and print the result

echo "Triggering AI/ML Job Market Pipeline..."
curl -s -X POST http://localhost:8000/run-pipeline | python3 -m json.tool
