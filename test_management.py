import sys
from src.management_governance import ManagementGovernanceAnalyzer

if __name__ == '__main__':
    print("Testing Management Governance Analyzer (which uses the earnings calls)...")
    analyzer = ManagementGovernanceAnalyzer()
    result = analyzer.analyze("AAPL")
    print(result)
