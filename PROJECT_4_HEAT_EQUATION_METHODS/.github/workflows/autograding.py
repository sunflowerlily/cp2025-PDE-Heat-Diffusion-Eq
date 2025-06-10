#!/usr/bin/env python3
"""
Automated grading script for GitHub Classroom
Heat Equation Methods Project
"""

import os
import sys
import json
import subprocess
import unittest
import traceback
from io import StringIO
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class AutoGrader:
    def __init__(self):
        self.results = {"tests": []}
        self.total_score = 0
        self.max_score = 100
        
    def run_project_tests(self, test_file_path, max_points=100):
        """Run tests for the heat equation methods project with timeout"""
        
        try:
            # Set timeout for the entire test suite
            result = subprocess.run(
                [sys.executable, '-m', 'unittest', test_file_path, '-v'],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
                cwd=os.path.dirname(test_file_path)
            )
            
            # Parse test results
            output = result.stdout + result.stderr
            
            if result.returncode == 0:
                # All tests passed
                score = max_points
                status = "passed"
                message = "All tests passed successfully"
            else:
                # Calculate partial scores based on test output
                score = self._calculate_partial_score(output, max_points)
                status = "partial" if score > 0 else "failed"
                message = f"Partial credit: {score}/{max_points} points"
                
        except subprocess.TimeoutExpired:
            score = 0
            status = "timeout"
            message = "Tests timed out after 5 minutes"
            output = "Test execution timed out"
            
        except Exception as e:
            score = 0
            status = "error"
            message = f"Error running tests: {str(e)}"
            output = traceback.format_exc()
        
        self.results["tests"].append({
            "name": "Heat Equation Methods",
            "score": score,
            "max_score": max_points,
            "status": status,
            "message": message,
            "output": output[:2000]  # Limit output length
        })
        
        self.total_score = score
        
    def _calculate_partial_score(self, output, max_points):
        """Calculate partial scores based on test output"""
        
        # Count individual test results
        lines = output.split('\n')
        
        # Look for test method results
        test_scores = {
            'test_initialization_5pts': 5,
            'test_explicit_method_basic_15pts': 15,
            'test_implicit_method_basic_15pts': 15,
            'test_crank_nicolson_method_basic_15pts': 15,
            'test_solve_ivp_method_basic_15pts': 15,
            'test_heat_equation_ode_helper_10pts': 10,
            'test_compare_methods_10pts': 10,
            'test_physical_behavior_10pts': 10,
            'test_stability_explicit_5pts': 5,
            'test_error_handling_5pts': 5,
            'test_complete_workflow_5pts': 5
        }
        
        total_earned = 0
        
        for line in lines:
            for test_name, points in test_scores.items():
                if test_name in line and 'ok' in line:
                    total_earned += points
                elif test_name in line and ('FAIL' in line or 'ERROR' in line):
                    # Check if it's a NotImplementedError (student hasn't implemented)
                    if 'NotImplementedError' in output:
                        # Give 0 points for unimplemented functions
                        pass
                    else:
                        # Give partial credit for attempted implementation
                        total_earned += points // 2
        
        return min(total_earned, max_points)
        
    def generate_report(self):
        """Generate JSON report for GitHub Classroom"""
        self.results["score"] = self.total_score
        self.results["max_score"] = self.max_score
        self.results["timestamp"] = datetime.now().isoformat()
        
        # Write results to file
        with open("autograding_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
            
        # Print summary
        print(f"\n{'='*50}")
        print(f"AUTOGRADING RESULTS")
        print(f"{'='*50}")
        print(f"Total Score: {self.total_score}/{self.max_score}")
        print(f"Percentage: {(self.total_score/self.max_score)*100:.1f}%")
        
        if self.total_score == self.max_score:
            print("🎉 Excellent work! All tests passed!")
        elif self.total_score >= self.max_score * 0.8:
            print("✅ Good work! Most functionality implemented correctly.")
        elif self.total_score >= self.max_score * 0.6:
            print("⚠️  Partial implementation. Some methods need work.")
        else:
            print("❌ Significant issues found. Please review implementation.")
            
        print(f"{'='*50}\n")
        
        return self.results

def main():
    """Main grading function"""
    print("Starting automated grading for Heat Equation Methods project...")
    
    # Initialize grader
    grader = AutoGrader()
    
    # Path to test file
    test_file = "tests/test_heat_equation_methods.py"
    
    # Check if test file exists
    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found!")
        sys.exit(1)
    
    # Check if student file exists
    student_file = "heat_equation_methods_student.py"
    if not os.path.exists(student_file):
        print(f"Error: Student file {student_file} not found!")
        sys.exit(1)
    
    # Run tests
    grader.run_project_tests(test_file, max_points=100)
    
    # Generate report
    results = grader.generate_report()
    
    # Exit with appropriate code
    if grader.total_score == grader.max_score:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Partial or failed

if __name__ == "__main__":
    main()