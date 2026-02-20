#!/bin/bash

echo "🔧 Setting up environment..."

pip install -r requirements.txt

echo "🚀 Running project..."

cd src
python main.py