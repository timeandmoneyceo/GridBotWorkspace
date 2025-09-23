#!/usr/bin/env python3
"""
Quick test to verify timeout mechanisms work
"""
import time
import threading
import requests

def test_ollama_connection():
    """Test Ollama connection with timeout"""
    print("Testing Ollama connection...")
    
    def make_request():
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen3:1.7b",
                    "prompt": "Hello, this is a test.",
                    "stream": False
                },
                timeout=30
            )
            print(f"Response status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {result.get('response', 'No response')[:100]}...")
            else:
                print(f"Error: {response.text}")
        except requests.exceptions.Timeout:
            print("Request timed out after 30 seconds")
        except Exception as e:
            print(f"Error: {e}")
    
    # Test with thread timeout
    result_container = {'done': False}
    
    def worker():
        make_request()
        result_container['done'] = True
    
    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    worker_thread.join(timeout=45)  # 45 second overall timeout
    
    if worker_thread.is_alive():
        print("Thread timeout - request took too long")
        return False
    elif result_container['done']:
        print("Request completed successfully")
        return True
    else:
        print("Request failed")
        return False

if __name__ == "__main__":
    success = test_ollama_connection()
    print(f"Test result: {'PASS' if success else 'FAIL'}")