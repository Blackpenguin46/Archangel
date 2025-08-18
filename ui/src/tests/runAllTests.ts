#!/usr/bin/env node

/**
 * Test runner for UI components
 * Runs all UI tests and generates a comprehensive report
 */

import { execSync } from 'child_process';
import { writeFileSync } from 'fs';
import path from 'path';

interface TestResult {
  testFile: string;
  passed: boolean;
  duration: number;
  coverage?: number;
  errors?: string[];
}

interface TestSuite {
  name: string;
  description: string;
  testFiles: string[];
}

const testSuites: TestSuite[] = [
  {
    name: 'Dashboard Tests',
    description: 'Tests for the main dashboard interface and real-time monitoring',
    testFiles: ['Dashboard.test.tsx'],
  },
  {
    name: 'Scenario Management Tests',
    description: 'Tests for scenario creation, configuration, and management',
    testFiles: ['Scenarios.test.tsx', 'ScenarioDetails.test.tsx'],
  },
  {
    name: 'Agent Management Tests',
    description: 'Tests for agent monitoring, configuration, and activity visualization',
    testFiles: ['Agents.test.tsx'],
  },
  {
    name: 'Network Topology Tests',
    description: 'Tests for network visualization and topology display',
    testFiles: ['NetworkTopology.test.tsx'],
  },
  {
    name: 'System Context Tests',
    description: 'Tests for system state management and data flow',
    testFiles: ['SystemContext.test.tsx'],
  },
  {
    name: 'UI Responsiveness Tests',
    description: 'Tests for UI performance, responsiveness, and data accuracy',
    testFiles: ['UIResponsiveness.test.tsx'],
  },
];

class TestRunner {
  private results: TestResult[] = [];
  private startTime: number = 0;
  private endTime: number = 0;

  async runAllTests(): Promise<void> {
    console.log('🚀 Starting UI Test Suite...\n');
    this.startTime = Date.now();

    for (const suite of testSuites) {
      console.log(`📋 Running ${suite.name}...`);
      console.log(`   ${suite.description}\n`);

      for (const testFile of suite.testFiles) {
        await this.runSingleTest(testFile);
      }
      
      console.log(''); // Add spacing between suites
    }

    this.endTime = Date.now();
    this.generateReport();
  }

  private async runSingleTest(testFile: string): Promise<void> {
    const startTime = Date.now();
    
    try {
      console.log(`   ⏳ Running ${testFile}...`);
      
      // Run the test with Jest
      const command = `npm test -- --testPathPattern=${testFile} --watchAll=false --coverage --silent`;
      const output = execSync(command, { 
        cwd: process.cwd(),
        encoding: 'utf8',
        stdio: 'pipe'
      });
      
      const duration = Date.now() - startTime;
      
      // Parse coverage from output (simplified)
      const coverageMatch = output.match(/All files\s+\|\s+([\d.]+)/);
      const coverage = coverageMatch ? parseFloat(coverageMatch[1]) : undefined;
      
      this.results.push({
        testFile,
        passed: true,
        duration,
        coverage,
      });
      
      console.log(`   ✅ ${testFile} passed (${duration}ms)`);
      if (coverage) {
        console.log(`      Coverage: ${coverage}%`);
      }
      
    } catch (error) {
      const duration = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : String(error);
      
      this.results.push({
        testFile,
        passed: false,
        duration,
        errors: [errorMessage],
      });
      
      console.log(`   ❌ ${testFile} failed (${duration}ms)`);
      console.log(`      Error: ${errorMessage.split('\n')[0]}`);
    }
  }

  private generateReport(): void {
    const totalTests = this.results.length;
    const passedTests = this.results.filter(r => r.passed).length;
    const failedTests = totalTests - passedTests;
    const totalDuration = this.endTime - this.startTime;
    const avgCoverage = this.calculateAverageCoverage();

    const report = {
      summary: {
        totalTests,
        passedTests,
        failedTests,
        successRate: (passedTests / totalTests) * 100,
        totalDuration,
        avgCoverage,
        timestamp: new Date().toISOString(),
      },
      testResults: this.results,
      recommendations: this.generateRecommendations(),
    };

    // Write detailed report to file
    const reportPath = path.join(process.cwd(), 'test-report.json');
    writeFileSync(reportPath, JSON.stringify(report, null, 2));

    // Print summary to console
    this.printSummary(report);
  }

  private calculateAverageCoverage(): number {
    const coverageResults = this.results.filter(r => r.coverage !== undefined);
    if (coverageResults.length === 0) return 0;
    
    const totalCoverage = coverageResults.reduce((sum, r) => sum + (r.coverage || 0), 0);
    return totalCoverage / coverageResults.length;
  }

  private generateRecommendations(): string[] {
    const recommendations: string[] = [];
    
    const failedTests = this.results.filter(r => !r.passed);
    if (failedTests.length > 0) {
      recommendations.push(`Fix ${failedTests.length} failing test(s)`);
    }
    
    const avgCoverage = this.calculateAverageCoverage();
    if (avgCoverage < 80) {
      recommendations.push(`Improve test coverage (currently ${avgCoverage.toFixed(1)}%, target: 80%)`);
    }
    
    const slowTests = this.results.filter(r => r.duration > 5000);
    if (slowTests.length > 0) {
      recommendations.push(`Optimize ${slowTests.length} slow test(s) (>5s execution time)`);
    }
    
    if (recommendations.length === 0) {
      recommendations.push('All tests are passing with good coverage and performance! 🎉');
    }
    
    return recommendations;
  }

  private printSummary(report: any): void {
    console.log('\n' + '='.repeat(60));
    console.log('📊 UI TEST SUITE SUMMARY');
    console.log('='.repeat(60));
    
    console.log(`\n📈 Results:`);
    console.log(`   Total Tests: ${report.summary.totalTests}`);
    console.log(`   Passed: ${report.summary.passedTests} ✅`);
    console.log(`   Failed: ${report.summary.failedTests} ${report.summary.failedTests > 0 ? '❌' : '✅'}`);
    console.log(`   Success Rate: ${report.summary.successRate.toFixed(1)}%`);
    
    console.log(`\n⏱️  Performance:`);
    console.log(`   Total Duration: ${report.summary.totalDuration}ms`);
    console.log(`   Average per Test: ${(report.summary.totalDuration / report.summary.totalTests).toFixed(0)}ms`);
    
    if (report.summary.avgCoverage > 0) {
      console.log(`\n📋 Coverage:`);
      console.log(`   Average Coverage: ${report.summary.avgCoverage.toFixed(1)}%`);
    }
    
    console.log(`\n💡 Recommendations:`);
    report.recommendations.forEach((rec: string) => {
      console.log(`   • ${rec}`);
    });
    
    if (report.summary.failedTests > 0) {
      console.log(`\n❌ Failed Tests:`);
      this.results.filter(r => !r.passed).forEach(result => {
        console.log(`   • ${result.testFile}`);
        if (result.errors) {
          result.errors.forEach(error => {
            console.log(`     ${error.split('\n')[0]}`);
          });
        }
      });
    }
    
    console.log(`\n📄 Detailed report saved to: test-report.json`);
    console.log('='.repeat(60));
    
    // Exit with appropriate code
    process.exit(report.summary.failedTests > 0 ? 1 : 0);
  }
}

// Performance benchmarks
const performanceBenchmarks = {
  maxTestDuration: 5000, // 5 seconds
  minCoverage: 80, // 80%
  maxMemoryUsage: 512, // 512MB
  maxRenderTime: 100, // 100ms for component rendering
};

// Accessibility checks
const accessibilityChecks = [
  'ARIA labels are present',
  'Keyboard navigation works',
  'Color contrast is sufficient',
  'Screen reader compatibility',
  'Focus management is correct',
];

// Run the tests if this file is executed directly
if (require.main === module) {
  const runner = new TestRunner();
  runner.runAllTests().catch(error => {
    console.error('❌ Test runner failed:', error);
    process.exit(1);
  });
}

export { TestRunner, performanceBenchmarks, accessibilityChecks };