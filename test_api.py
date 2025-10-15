#!/usr/bin/env python3
"""
Simple test script for the Chinese Legal RAG Text Generation API
"""

import requests
import json
import time
import sys

API_BASE = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            print(f"   Model type: {data.get('model_type')}")
            print(f"   Vocab size: {data.get('vocab_size')}")
            print(f"   Device: {data.get('device')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_generation():
    """Test text generation endpoint"""
    print("\n🔍 Testing text generation...")
    
    test_prompt = "王军的行为是否符合中国刑法关于盗窃罪的构成要件"
    
    data = {
        "prompt": test_prompt,
        "max_length": 50
    }
    
    try:
        print(f"   Prompt: {test_prompt}")
        print("   Generating...")
        
        start_time = time.time()
        response = requests.post(f"{API_BASE}/generate", json=data, timeout=30)
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Generation successful ({generation_time:.2f}s)")
            print(f"   Generated text: {result.get('generated_text', '')[:100]}...")
            print(f"   Model type: {result.get('model_type')}")
            return True
        else:
            print(f"❌ Generation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Generation failed: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\n🔍 Testing model info endpoint...")
    try:
        response = requests.get(f"{API_BASE}/model-info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved")
            print(f"   Model type: {data.get('model_type')}")
            print(f"   Total parameters: {data.get('total_parameters'):,}")
            print(f"   Trainable parameters: {data.get('trainable_parameters'):,}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Model info failed: {e}")
        return False

def test_sample_prompts():
    """Test sample prompts endpoint"""
    print("\n🔍 Testing sample prompts...")
    try:
        response = requests.post(f"{API_BASE}/test-prompts", timeout=60)
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Sample prompts test passed")
            print(f"   Generated {len(results)} responses")
            for i, result in enumerate(results[:2]):  # Show first 2
                print(f"   Sample {i+1}: {result.get('generated_text', '')[:50]}...")
            return True
        else:
            print(f"❌ Sample prompts failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Sample prompts failed: {e}")
        return False

def main():
    print("🧪 Chinese Legal RAG API Test Suite")
    print("=" * 40)
    
    # Check if server is running
    print(f"🌐 Testing API at: {API_BASE}")
    
    try:
        requests.get(f"{API_BASE}/", timeout=5)
    except requests.exceptions.RequestException:
        print("❌ API server not reachable!")
        print("   Make sure the server is running:")
        print("   python run_api.py")
        sys.exit(1)
    
    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Text Generation", test_generation),
        ("Model Info", test_model_info),
        ("Sample Prompts", test_sample_prompts),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 40)
    print(f"🏁 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
