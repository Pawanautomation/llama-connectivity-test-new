# src/run_test.py
from llama_test.core import LlamaConnectivityTest
import logging

logger = logging.getLogger(__name__)

def main():
    """Run LLaMA connectivity tests."""
    print("\n=== LLaMA Connectivity Test ===\n")
    
    # Initialize and run tests
    tester = LlamaConnectivityTest()
    results = tester.run_all_tests()
    
    # Display results
    print("\n=== Test Results ===")
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name:15}: {status}")
    
    # Overall status
    if all(results.values()):
        print("\n✅ All tests passed! LLaMA integration is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the logs for details.")
        print(f"Log file location: {tester.LOG_FILE}")

if __name__ == "__main__":
    main()