#!/usr/bin/env python3
"""
Automated grading script for GitHub Classroom
Project: Heat Equation Methods Comparison
"""

import os
import sys
import json
import subprocess
import unittest
from io import StringIO
import traceback

class AutoGrader:
    def __init__(self):
        self.results = {"tests": []}
        self.total_score = 0
        self.max_score = 100
        
    def run_project_tests(self, project_name="heat_equation_methods", max_points=100):
        """Run tests for the heat equation methods project"""
        print(f"\n=== Running tests for {project_name} ===")
        
        # Test categories with points
        test_categories = [
            ("test_reference_solution", 0, "Reference solution validation"),
            ("test_student_basic_implementation_20pts", 20, "Basic implementation"),
            ("test_student_method_accuracy_25pts", 25, "Method accuracy"),
            ("test_student_comparison_analysis_15pts", 15, "Comparison analysis"),
            ("test_student_plotting_functionality_10pts", 10, "Plotting functionality"),
            ("test_student_error_handling_10pts", 10, "Error handling"),
            ("test_student_boundary_conditions_10pts", 10, "Boundary conditions"),
            ("test_student_stability_awareness_5pts", 5, "Stability awareness"),
            ("test_student_code_quality_5pts", 5, "Code quality")
        ]
        
        # Import test module
        try:
            sys.path.insert(0, 'tests')
            from test_heat_equation_methods import TestHeatEquationMethods
            
            # Run each test category
            for test_method, points, description in test_categories:
                try:
                    suite = unittest.TestSuite()
                    test_case = TestHeatEquationMethods(test_method)
                    suite.addTest(test_case)
                    
                    # Capture test output
                    stream = StringIO()
                    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
                    result = runner.run(suite)
                    
                    # Determine score
                    if result.wasSuccessful():
                        score = points
                        status = "passed"
                        output = f"✓ {description} - All tests passed"
                    else:
                        score = 0
                        status = "failed"
                        errors = []
                        for failure in result.failures + result.errors:
                            errors.append(failure[1])
                        output = f"✗ {description} - {'; '.join(errors[:2])}"
                    
                    self.results["tests"].append({
                        "name": description,
                        "status": status,
                        "message": output,
                        "test_code": test_method,
                        "filename": "test_heat_equation_methods.py",
                        "line_no": 1,
                        "execution_time": "0.1s"
                    })
                    
                    self.total_score += score
                    print(f"{description}: {score}/{points} points")
                    
                except Exception as e:
                    error_msg = f"Error running {test_method}: {str(e)}"
                    print(error_msg)
                    self.results["tests"].append({
                        "name": description,
                        "status": "error",
                        "message": error_msg,
                        "test_code": test_method,
                        "filename": "test_heat_equation_methods.py",
                        "line_no": 1,
                        "execution_time": "0.1s"
                    })
                    
        except ImportError as e:
            error_msg = f"Failed to import test module: {str(e)}"
            print(error_msg)
            self.results["tests"].append({
                "name": "Import Tests",
                "status": "error",
                "message": error_msg,
                "test_code": "import_test",
                "filename": "test_heat_equation_methods.py",
                "line_no": 1,
                "execution_time": "0.1s"
            })
            
    def check_file_structure(self):
        """Check if required files exist"""
        required_files = [
            "heat_equation_methods_student.py",
            "实验报告模板.md",
            "requirements.txt"
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
                
        if missing_files:
            self.results["tests"].append({
                "name": "File Structure Check",
                "status": "failed",
                "message": f"Missing files: {', '.join(missing_files)}",
                "test_code": "file_structure",
                "filename": "autograding.py",
                "line_no": 1,
                "execution_time": "0.1s"
            })
        else:
            self.results["tests"].append({
                "name": "File Structure Check",
                "status": "passed",
                "message": "All required files present",
                "test_code": "file_structure",
                "filename": "autograding.py",
                "line_no": 1,
                "execution_time": "0.1s"
            })
            
    def generate_report(self):
        """Generate JSON report for GitHub Classroom"""
        print(f"\n=== Grading Summary ===")
        print(f"Total Score: {self.total_score}/{self.max_score}")
        print(f"Percentage: {(self.total_score/self.max_score)*100:.1f}%")
        
        # Add summary to results
        self.results["score"] = self.total_score
        self.results["max_score"] = self.max_score
        self.results["percentage"] = (self.total_score/self.max_score)*100
        
        # Write results to file for GitHub Classroom
        with open('autograding_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
            
        print("\nDetailed results saved to autograding_results.json")
        
        # Print test summary
        print("\n=== Test Results ===")
        for test in self.results["tests"]:
            status_symbol = "✓" if test["status"] == "passed" else "✗" if test["status"] == "failed" else "!"
            print(f"{status_symbol} {test['name']}: {test['message']}")
            
if __name__ == "__main__":
    print("GitHub Classroom Autograder")
    print("Project: Heat Equation Methods Comparison")
    print("=" * 50)
    
    grader = AutoGrader()
    
    # Check file structure
    grader.check_file_structure()
    
    # Run project tests
    grader.run_project_tests()
    
    # Generate final report
    grader.generate_report()
    
    print("\nAutograding completed!")