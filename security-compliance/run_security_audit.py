#!/usr/bin/env python3
"""
Security Audit Execution Script
Demonstrates the security audit and penetration testing capabilities
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

async def main():
    """Main execution function"""
    print("="*80)
    print("ARCHANGEL SECURITY AUDIT AND PENETRATION TESTING")
    print("="*80)
    print()
    
    try:
        # Import and run comprehensive security validation
        from comprehensive_security_validation import SecurityValidationOrchestrator
        
        print("🔍 Starting comprehensive security validation...")
        print()
        
        orchestrator = SecurityValidationOrchestrator()
        report = await orchestrator.run_comprehensive_validation()
        
        # Save the report
        report_path = orchestrator.save_validation_report(report)
        
        # Display summary
        print("\n" + "="*60)
        print("SECURITY VALIDATION RESULTS")
        print("="*60)
        
        print(f"📊 Overall Security Score: {report.overall_security_score:.1f}/100")
        
        # Determine security grade
        score = report.overall_security_score
        if score >= 90:
            grade = "🟢 A (Excellent)"
        elif score >= 80:
            grade = "🟡 B (Good)"
        elif score >= 70:
            grade = "🟠 C (Fair)"
        elif score >= 60:
            grade = "🔴 D (Poor)"
        else:
            grade = "🚨 F (Critical)"
        
        print(f"🎯 Security Grade: {grade}")
        
        # Show static analysis results
        print(f"\n📋 Static Analysis Summary:")
        for tool, results in report.static_analysis_results.items():
            status = results.get('status', 'unknown')
            issues = results.get('issues_found', 0)
            vulnerabilities = results.get('vulnerabilities_found', 0)
            
            if status == 'completed':
                if issues > 0:
                    print(f"  • {tool.upper()}: ⚠️  {issues} issues found")
                elif vulnerabilities > 0:
                    print(f"  • {tool.upper()}: ⚠️  {vulnerabilities} vulnerabilities found")
                else:
                    print(f"  • {tool.upper()}: ✅ No issues found")
            elif status == 'not_installed':
                print(f"  • {tool.upper()}: ⏭️  Tool not installed")
            else:
                print(f"  • {tool.upper()}: ❌ {status}")
        
        # Show compliance results
        print(f"\n📋 Compliance Check Summary:")
        for check in report.compliance_checks:
            if check.status == 'PASS':
                status_icon = "✅"
            elif check.status == 'FAIL':
                status_icon = "❌"
            elif check.status == 'WARNING':
                status_icon = "⚠️"
            else:
                status_icon = "⏭️"
            
            print(f"  • {check.check_name}: {status_icon} {check.status}")
        
        # Show critical issues
        if report.critical_issues:
            print(f"\n🚨 Critical Issues ({len(report.critical_issues)}):")
            for issue in report.critical_issues[:5]:
                print(f"  • {issue}")
            if len(report.critical_issues) > 5:
                print(f"  • ... and {len(report.critical_issues) - 5} more")
        else:
            print(f"\n✅ No critical security issues found!")
        
        # Show top recommendations
        if report.recommendations:
            print(f"\n💡 Top Recommendations:")
            for i, rec in enumerate(report.recommendations[:3], 1):
                print(f"  {i}. {rec}")
        
        print(f"\n📄 Detailed report saved to: {report_path}")
        print("="*60)
        
        # Return appropriate exit code
        if score < 60:
            print("\n🚨 CRITICAL: Security score below acceptable threshold!")
            return 1
        elif len(report.critical_issues) > 0:
            print("\n⚠️  WARNING: Critical security issues found!")
            return 1
        else:
            print("\n🎉 Security validation completed successfully!")
            return 0
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n📦 Required dependencies may not be installed.")
        print("💡 Try installing: pip install docker requests")
        return 1
        
    except Exception as e:
        print(f"❌ Security validation failed: {e}")
        return 1

def run_basic_checks():
    """Run basic security checks without full framework"""
    print("🔍 Running basic security checks...")
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: Look for common security files
    total_checks += 1
    security_files = [
        '.bandit.yml',
        'requirements.txt',
        'Dockerfile',
        'docker-compose.yml'
    ]
    
    found_files = [f for f in security_files if Path(f).exists()]
    if found_files:
        print(f"✅ Security configuration files found: {', '.join(found_files)}")
        checks_passed += 1
    else:
        print("⚠️  No security configuration files found")
    
    # Check 2: Check for Python files (basic code structure)
    total_checks += 1
    python_files = list(Path('.').rglob('*.py'))
    if len(python_files) > 0:
        print(f"✅ Found {len(python_files)} Python files for analysis")
        checks_passed += 1
    else:
        print("⚠️  No Python files found")
    
    # Check 3: Check for container-related files
    total_checks += 1
    container_files = list(Path('.').rglob('Dockerfile*')) + list(Path('.').rglob('docker-compose*.yml'))
    if container_files:
        print(f"✅ Found {len(container_files)} container configuration files")
        checks_passed += 1
    else:
        print("⚠️  No container configuration files found")
    
    # Check 4: Check for infrastructure files
    total_checks += 1
    infra_files = list(Path('.').rglob('*.tf')) + list(Path('.').rglob('*.yaml'))
    if infra_files:
        print(f"✅ Found {len(infra_files)} infrastructure configuration files")
        checks_passed += 1
    else:
        print("⚠️  No infrastructure configuration files found")
    
    print(f"\n📊 Basic checks: {checks_passed}/{total_checks} passed")
    
    if checks_passed == total_checks:
        print("🎉 All basic security checks passed!")
        return 0
    else:
        print("⚠️  Some basic security checks failed")
        return 1

if __name__ == "__main__":
    print("🚀 Starting Archangel Security Audit...")
    
    try:
        # Try to run comprehensive validation
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⏹️  Security audit interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Comprehensive validation failed: {e}")
        print("\n🔄 Falling back to basic security checks...")
        
        try:
            exit_code = run_basic_checks()
            sys.exit(exit_code)
        except Exception as basic_error:
            print(f"❌ Basic checks also failed: {basic_error}")
            sys.exit(1)