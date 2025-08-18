#!/usr/bin/env python3
"""
Compliance Framework for Security Standards
NIST Cybersecurity Framework, ISO 27001, and other security standards compliance
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from collections import defaultdict
import yaml
from abc import ABC, abstractmethod


class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant" 
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_ASSESSED = "not_assessed"
    EXEMPT = "exempt"


class ComplianceFramework(Enum):
    NIST_CSF = "nist_csf"
    ISO_27001 = "iso_27001"
    SOC2 = "soc2"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    GDPR = "gdpr"


class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ComplianceRequirement:
    requirement_id: str
    framework: ComplianceFramework
    title: str
    description: str
    category: str
    subcategory: Optional[str] = None
    risk_level: RiskLevel = RiskLevel.MEDIUM
    status: ComplianceStatus = ComplianceStatus.NOT_ASSESSED
    implementation_guidance: List[str] = field(default_factory=list)
    evidence_requirements: List[str] = field(default_factory=list)
    automated_checks: List[str] = field(default_factory=list)
    last_assessment: Optional[datetime] = None
    next_review: Optional[datetime] = None
    assigned_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceEvidence:
    evidence_id: str
    requirement_id: str
    evidence_type: str  # document, screenshot, log, certificate, etc.
    title: str
    description: str
    file_path: Optional[str] = None
    content: Optional[str] = None
    collected_date: datetime = field(default_factory=datetime.now)
    expiry_date: Optional[datetime] = None
    validated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceAssessment:
    assessment_id: str
    framework: ComplianceFramework
    assessment_date: datetime
    assessor: str
    scope: List[str]
    overall_status: ComplianceStatus
    compliance_percentage: float
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    next_assessment: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComplianceFrameworkBase(ABC):
    """Abstract base class for compliance frameworks"""
    
    @abstractmethod
    def get_requirements(self) -> List[ComplianceRequirement]:
        """Get all requirements for this framework"""
        pass
    
    @abstractmethod
    def assess_requirement(self, requirement: ComplianceRequirement) -> ComplianceStatus:
        """Assess compliance status for a specific requirement"""
        pass
    
    @abstractmethod
    def generate_gap_analysis(self, requirements: List[ComplianceRequirement]) -> Dict[str, Any]:
        """Generate gap analysis report"""
        pass


class NISTCybersecurityFramework(ComplianceFrameworkBase):
    """NIST Cybersecurity Framework implementation"""
    
    def get_requirements(self) -> List[ComplianceRequirement]:
        """Get NIST CSF requirements"""
        return [
            # IDENTIFY (ID)
            ComplianceRequirement(
                requirement_id="ID.AM-1",
                framework=ComplianceFramework.NIST_CSF,
                title="Asset Management",
                description="Physical devices and systems within the organization are inventoried",
                category="Identify",
                subcategory="Asset Management",
                risk_level=RiskLevel.HIGH,
                implementation_guidance=[
                    "Maintain comprehensive asset inventory",
                    "Include all physical devices, systems, and software",
                    "Regular updates and validation of asset information"
                ],
                evidence_requirements=[
                    "Asset inventory database",
                    "Asset discovery scan results",
                    "Asset management procedures"
                ],
                automated_checks=["asset_inventory_scan", "network_discovery"]
            ),
            ComplianceRequirement(
                requirement_id="ID.AM-2", 
                framework=ComplianceFramework.NIST_CSF,
                title="Software Asset Management",
                description="Software platforms and applications within the organization are inventoried",
                category="Identify",
                subcategory="Asset Management",
                risk_level=RiskLevel.HIGH,
                implementation_guidance=[
                    "Catalog all software applications",
                    "Track software versions and licenses",
                    "Monitor unauthorized software installation"
                ],
                evidence_requirements=[
                    "Software inventory report",
                    "License management records",
                    "Software deployment policies"
                ],
                automated_checks=["software_inventory_scan", "license_compliance_check"]
            ),
            ComplianceRequirement(
                requirement_id="ID.GV-1",
                framework=ComplianceFramework.NIST_CSF,
                title="Information Security Policy",
                description="Organizational cybersecurity policy is established and communicated",
                category="Identify", 
                subcategory="Governance",
                risk_level=RiskLevel.HIGH,
                implementation_guidance=[
                    "Develop comprehensive cybersecurity policy",
                    "Ensure senior management approval",
                    "Communicate policy to all stakeholders"
                ],
                evidence_requirements=[
                    "Cybersecurity policy document",
                    "Management approval records",
                    "Policy communication evidence"
                ],
                automated_checks=["policy_document_check", "policy_distribution_verification"]
            ),
            
            # PROTECT (PR)
            ComplianceRequirement(
                requirement_id="PR.AC-1",
                framework=ComplianceFramework.NIST_CSF,
                title="Access Control Management",
                description="Identities and credentials are issued, managed, verified, revoked, and audited",
                category="Protect",
                subcategory="Access Control",
                risk_level=RiskLevel.CRITICAL,
                implementation_guidance=[
                    "Implement identity and access management system",
                    "Regular access reviews and provisioning",
                    "Strong authentication mechanisms"
                ],
                evidence_requirements=[
                    "Access control matrix",
                    "User access review reports",
                    "Authentication system logs"
                ],
                automated_checks=["access_review_audit", "authentication_strength_check"]
            ),
            ComplianceRequirement(
                requirement_id="PR.DS-1",
                framework=ComplianceFramework.NIST_CSF,
                title="Data Protection in Transit",
                description="Data-in-transit is protected",
                category="Protect",
                subcategory="Data Security",
                risk_level=RiskLevel.HIGH,
                implementation_guidance=[
                    "Implement encryption for data in transit",
                    "Use secure communication protocols",
                    "Monitor and validate encryption implementation"
                ],
                evidence_requirements=[
                    "Encryption configuration documentation",
                    "Network traffic analysis",
                    "SSL/TLS certificate inventory"
                ],
                automated_checks=["encryption_in_transit_scan", "ssl_certificate_check"]
            ),
            
            # DETECT (DE)
            ComplianceRequirement(
                requirement_id="DE.AE-1",
                framework=ComplianceFramework.NIST_CSF,
                title="Event Detection",
                description="A baseline of network operations is established and managed",
                category="Detect",
                subcategory="Anomalies and Events",
                risk_level=RiskLevel.HIGH,
                implementation_guidance=[
                    "Establish network baseline",
                    "Implement continuous monitoring",
                    "Define normal vs. anomalous behavior"
                ],
                evidence_requirements=[
                    "Network baseline documentation",
                    "Monitoring system configuration",
                    "Anomaly detection reports"
                ],
                automated_checks=["baseline_establishment", "anomaly_detection_verification"]
            ),
            
            # RESPOND (RS)
            ComplianceRequirement(
                requirement_id="RS.RP-1",
                framework=ComplianceFramework.NIST_CSF,
                title="Response Planning",
                description="Response plan is executed during or after an incident",
                category="Respond",
                subcategory="Response Planning", 
                risk_level=RiskLevel.HIGH,
                implementation_guidance=[
                    "Develop incident response plan",
                    "Define roles and responsibilities",
                    "Regular testing and updates"
                ],
                evidence_requirements=[
                    "Incident response plan document",
                    "Response team contact list",
                    "Incident response testing records"
                ],
                automated_checks=["response_plan_validation", "team_readiness_check"]
            ),
            
            # RECOVER (RC)
            ComplianceRequirement(
                requirement_id="RC.RP-1",
                framework=ComplianceFramework.NIST_CSF,
                title="Recovery Planning",
                description="Recovery plan is executed during or after a cybersecurity incident",
                category="Recover",
                subcategory="Recovery Planning",
                risk_level=RiskLevel.HIGH,
                implementation_guidance=[
                    "Develop recovery procedures",
                    "Define recovery priorities",
                    "Regular backup and restore testing"
                ],
                evidence_requirements=[
                    "Recovery plan documentation",
                    "Backup and restore procedures",
                    "Recovery testing results"
                ],
                automated_checks=["backup_verification", "recovery_procedure_test"]
            )
        ]
    
    def assess_requirement(self, requirement: ComplianceRequirement) -> ComplianceStatus:
        """Assess NIST CSF requirement compliance"""
        # This would integrate with actual security controls and evidence
        # For now, return sample assessment based on requirement type
        
        if requirement.requirement_id.startswith("ID.AM"):
            # Asset Management - check if asset inventory exists
            return ComplianceStatus.COMPLIANT if self._check_asset_inventory() else ComplianceStatus.NON_COMPLIANT
        elif requirement.requirement_id.startswith("PR.AC"):
            # Access Control - check access management
            return ComplianceStatus.PARTIALLY_COMPLIANT  # Example
        else:
            return ComplianceStatus.NOT_ASSESSED
    
    def _check_asset_inventory(self) -> bool:
        """Check if asset inventory exists"""
        # Placeholder - would check actual asset management system
        return True
    
    def generate_gap_analysis(self, requirements: List[ComplianceRequirement]) -> Dict[str, Any]:
        """Generate NIST CSF gap analysis"""
        gaps = []
        compliant_count = 0
        
        for req in requirements:
            if req.status == ComplianceStatus.NON_COMPLIANT:
                gaps.append({
                    'requirement_id': req.requirement_id,
                    'title': req.title,
                    'category': req.category,
                    'risk_level': req.risk_level.value,
                    'implementation_guidance': req.implementation_guidance
                })
            elif req.status == ComplianceStatus.COMPLIANT:
                compliant_count += 1
        
        return {
            'framework': 'NIST Cybersecurity Framework',
            'total_requirements': len(requirements),
            'compliant_requirements': compliant_count,
            'compliance_percentage': (compliant_count / len(requirements)) * 100,
            'gaps': gaps,
            'recommendations': self._generate_nist_recommendations(gaps)
        }
    
    def _generate_nist_recommendations(self, gaps: List[Dict[str, Any]]) -> List[str]:
        """Generate NIST-specific recommendations"""
        recommendations = []
        
        gap_categories = defaultdict(int)
        for gap in gaps:
            gap_categories[gap['category']] += 1
        
        if gap_categories['Identify'] > 0:
            recommendations.append("Focus on asset discovery and risk assessment activities")
        if gap_categories['Protect'] > 0:
            recommendations.append("Strengthen protective controls and access management")
        if gap_categories['Detect'] > 0:
            recommendations.append("Enhance monitoring and detection capabilities")
        if gap_categories['Respond'] > 0:
            recommendations.append("Develop and test incident response procedures")
        if gap_categories['Recover'] > 0:
            recommendations.append("Improve recovery and business continuity planning")
        
        return recommendations


class ISO27001Framework(ComplianceFrameworkBase):
    """ISO 27001 implementation"""
    
    def get_requirements(self) -> List[ComplianceRequirement]:
        """Get ISO 27001 requirements"""
        return [
            ComplianceRequirement(
                requirement_id="A.5.1.1",
                framework=ComplianceFramework.ISO_27001,
                title="Information Security Policies",
                description="A set of policies for information security shall be defined",
                category="Organization of Information Security",
                risk_level=RiskLevel.HIGH,
                implementation_guidance=[
                    "Define comprehensive information security policies",
                    "Ensure management approval and support",
                    "Communicate policies to relevant personnel"
                ],
                evidence_requirements=[
                    "Information security policy document",
                    "Management approval records",
                    "Policy distribution evidence"
                ]
            ),
            ComplianceRequirement(
                requirement_id="A.8.1.1",
                framework=ComplianceFramework.ISO_27001,
                title="Inventory of Assets",
                description="Assets associated with information shall be identified",
                category="Asset Management",
                risk_level=RiskLevel.HIGH,
                implementation_guidance=[
                    "Maintain accurate asset inventory",
                    "Classify assets based on value and criticality",
                    "Regular review and update of asset information"
                ],
                evidence_requirements=[
                    "Asset register",
                    "Asset classification guidelines",
                    "Regular asset review reports"
                ]
            ),
            ComplianceRequirement(
                requirement_id="A.9.1.1",
                framework=ComplianceFramework.ISO_27001,
                title="Access Control Policy",
                description="An access control policy shall be established",
                category="Access Control",
                risk_level=RiskLevel.CRITICAL,
                implementation_guidance=[
                    "Define access control policy and procedures",
                    "Implement role-based access controls",
                    "Regular access reviews and maintenance"
                ],
                evidence_requirements=[
                    "Access control policy",
                    "User access matrix",
                    "Access review reports"
                ]
            )
        ]
    
    def assess_requirement(self, requirement: ComplianceRequirement) -> ComplianceStatus:
        """Assess ISO 27001 requirement compliance"""
        # Placeholder assessment logic
        return ComplianceStatus.NOT_ASSESSED
    
    def generate_gap_analysis(self, requirements: List[ComplianceRequirement]) -> Dict[str, Any]:
        """Generate ISO 27001 gap analysis"""
        return {
            'framework': 'ISO 27001',
            'assessment_method': 'Control-based assessment',
            'gaps': [],
            'recommendations': []
        }


class ComplianceFrameworkManager:
    """Main compliance framework management system"""
    
    def __init__(self, db_path: str = "security/data/compliance.db"):
        self.db_path = db_path
        self.frameworks: Dict[ComplianceFramework, ComplianceFrameworkBase] = {
            ComplianceFramework.NIST_CSF: NISTCybersecurityFramework(),
            ComplianceFramework.ISO_27001: ISO27001Framework()
        }
        self.requirements: Dict[str, ComplianceRequirement] = {}
        self.evidence: Dict[str, ComplianceEvidence] = {}
        self.assessments: List[ComplianceAssessment] = []
        
        self._init_database()
        self._load_requirements()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self) -> None:
        """Initialize compliance database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Requirements table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_requirements (
                requirement_id TEXT PRIMARY KEY,
                framework TEXT,
                title TEXT,
                description TEXT,
                category TEXT,
                subcategory TEXT,
                risk_level TEXT,
                status TEXT,
                last_assessment TEXT,
                next_review TEXT,
                assigned_to TEXT,
                metadata TEXT
            )
        ''')
        
        # Evidence table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_evidence (
                evidence_id TEXT PRIMARY KEY,
                requirement_id TEXT,
                evidence_type TEXT,
                title TEXT,
                description TEXT,
                file_path TEXT,
                collected_date TEXT,
                expiry_date TEXT,
                validated BOOLEAN,
                metadata TEXT,
                FOREIGN KEY (requirement_id) REFERENCES compliance_requirements (requirement_id)
            )
        ''')
        
        # Assessments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_assessments (
                assessment_id TEXT PRIMARY KEY,
                framework TEXT,
                assessment_date TEXT,
                assessor TEXT,
                scope TEXT,
                overall_status TEXT,
                compliance_percentage REAL,
                findings TEXT,
                recommendations TEXT,
                next_assessment TEXT,
                metadata TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_requirements(self) -> None:
        """Load requirements from all frameworks"""
        for framework, implementation in self.frameworks.items():
            requirements = implementation.get_requirements()
            for req in requirements:
                self.requirements[req.requirement_id] = req
    
    def assess_compliance(self, framework: ComplianceFramework, assessor: str) -> ComplianceAssessment:
        """Perform comprehensive compliance assessment"""
        framework_impl = self.frameworks.get(framework)
        if not framework_impl:
            raise ValueError(f"Framework {framework} not supported")
        
        # Get requirements for this framework
        requirements = [req for req in self.requirements.values() if req.framework == framework]
        
        # Assess each requirement
        compliant_count = 0
        findings = []
        
        for req in requirements:
            status = framework_impl.assess_requirement(req)
            req.status = status
            req.last_assessment = datetime.now()
            
            if status == ComplianceStatus.COMPLIANT:
                compliant_count += 1
            elif status == ComplianceStatus.NON_COMPLIANT:
                findings.append({
                    'requirement_id': req.requirement_id,
                    'title': req.title,
                    'finding': 'Non-compliant',
                    'risk_level': req.risk_level.value,
                    'recommendation': req.implementation_guidance
                })
        
        # Calculate compliance percentage
        compliance_percentage = (compliant_count / len(requirements)) * 100
        
        # Determine overall status
        if compliance_percentage >= 95:
            overall_status = ComplianceStatus.COMPLIANT
        elif compliance_percentage >= 70:
            overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            overall_status = ComplianceStatus.NON_COMPLIANT
        
        # Generate gap analysis
        gap_analysis = framework_impl.generate_gap_analysis(requirements)
        
        # Create assessment
        assessment = ComplianceAssessment(
            assessment_id=f"ASSESS_{framework.value}_{int(time.time())}",
            framework=framework,
            assessment_date=datetime.now(),
            assessor=assessor,
            scope=[req.requirement_id for req in requirements],
            overall_status=overall_status,
            compliance_percentage=compliance_percentage,
            findings=findings,
            recommendations=gap_analysis.get('recommendations', []),
            next_assessment=datetime.now() + timedelta(days=365)  # Annual assessment
        )
        
        self.assessments.append(assessment)
        self._save_assessment(assessment)
        
        return assessment
    
    def add_evidence(self, evidence: ComplianceEvidence) -> None:
        """Add compliance evidence"""
        self.evidence[evidence.evidence_id] = evidence
        self._save_evidence(evidence)
    
    def _save_assessment(self, assessment: ComplianceAssessment) -> None:
        """Save assessment to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO compliance_assessments 
            (assessment_id, framework, assessment_date, assessor, scope, overall_status,
             compliance_percentage, findings, recommendations, next_assessment, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            assessment.assessment_id,
            assessment.framework.value,
            assessment.assessment_date.isoformat(),
            assessment.assessor,
            json.dumps(assessment.scope),
            assessment.overall_status.value,
            assessment.compliance_percentage,
            json.dumps(assessment.findings),
            json.dumps(assessment.recommendations),
            assessment.next_assessment.isoformat() if assessment.next_assessment else None,
            json.dumps(assessment.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def _save_evidence(self, evidence: ComplianceEvidence) -> None:
        """Save evidence to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO compliance_evidence 
            (evidence_id, requirement_id, evidence_type, title, description, file_path,
             collected_date, expiry_date, validated, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            evidence.evidence_id,
            evidence.requirement_id,
            evidence.evidence_type,
            evidence.title,
            evidence.description,
            evidence.file_path,
            evidence.collected_date.isoformat(),
            evidence.expiry_date.isoformat() if evidence.expiry_date else None,
            evidence.validated,
            json.dumps(evidence.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def generate_compliance_dashboard(self) -> Dict[str, Any]:
        """Generate compliance dashboard data"""
        dashboard = {
            'last_updated': datetime.now().isoformat(),
            'frameworks': {},
            'overall_compliance': 0,
            'high_priority_gaps': [],
            'upcoming_reviews': []
        }
        
        total_compliance = 0
        framework_count = 0
        
        for framework in self.frameworks.keys():
            requirements = [req for req in self.requirements.values() if req.framework == framework]
            
            if requirements:
                compliant = len([req for req in requirements if req.status == ComplianceStatus.COMPLIANT])
                compliance_rate = (compliant / len(requirements)) * 100
                
                dashboard['frameworks'][framework.value] = {
                    'total_requirements': len(requirements),
                    'compliant': compliant,
                    'compliance_percentage': compliance_rate,
                    'last_assessment': max([req.last_assessment for req in requirements if req.last_assessment], default=None)
                }
                
                total_compliance += compliance_rate
                framework_count += 1
                
                # Find high-priority gaps
                for req in requirements:
                    if req.status == ComplianceStatus.NON_COMPLIANT and req.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
                        dashboard['high_priority_gaps'].append({
                            'requirement_id': req.requirement_id,
                            'title': req.title,
                            'framework': framework.value,
                            'risk_level': req.risk_level.value
                        })
        
        if framework_count > 0:
            dashboard['overall_compliance'] = total_compliance / framework_count
        
        return dashboard
    
    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary report"""
        dashboard = self.generate_compliance_dashboard()
        recent_assessments = self.assessments[-5:]  # Last 5 assessments
        
        return {
            'report_date': datetime.now().isoformat(),
            'executive_summary': {
                'overall_compliance_score': dashboard['overall_compliance'],
                'frameworks_assessed': len(dashboard['frameworks']),
                'high_priority_gaps': len(dashboard['high_priority_gaps']),
                'recent_assessments': len(recent_assessments)
            },
            'key_findings': [
                f"Overall compliance rate: {dashboard['overall_compliance']:.1f}%",
                f"High-priority gaps identified: {len(dashboard['high_priority_gaps'])}",
                f"Frameworks under management: {len(dashboard['frameworks'])}"
            ],
            'recommendations': [
                "Focus on high-priority compliance gaps",
                "Implement regular assessment schedule", 
                "Enhance evidence collection and documentation",
                "Strengthen control implementation and testing"
            ],
            'compliance_trends': dashboard,
            'next_actions': [
                "Schedule next compliance assessment",
                "Address high-priority findings",
                "Update policies and procedures",
                "Train staff on compliance requirements"
            ]
        }


# Example usage
if __name__ == "__main__":
    # Create compliance manager
    manager = ComplianceFrameworkManager()
    
    # Perform NIST CSF assessment
    print("Performing NIST Cybersecurity Framework assessment...")
    nist_assessment = manager.assess_compliance(ComplianceFramework.NIST_CSF, "Security Team")
    
    print(f"NIST CSF Compliance: {nist_assessment.compliance_percentage:.1f}%")
    print(f"Overall Status: {nist_assessment.overall_status.value}")
    print(f"Findings: {len(nist_assessment.findings)}")
    
    # Generate dashboard
    dashboard = manager.generate_compliance_dashboard()
    print(f"\nOverall Compliance Score: {dashboard['overall_compliance']:.1f}%")
    
    # Generate executive summary
    exec_summary = manager.generate_executive_summary()
    print(f"\nExecutive Summary Generated")
    print(f"Key Findings: {len(exec_summary['key_findings'])}")
    print(f"Recommendations: {len(exec_summary['recommendations'])}")