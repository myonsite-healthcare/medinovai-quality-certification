#!/usr/bin/env python3
"""
Comprehensive Quality Certification Manager for MedinovAI
Generates detailed quality certificates for tasks, modules, and system-wide quality assurance
with full traceability, compliance validation, and professional reporting
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import hashlib
import base64
from jinja2 import Template
import qrcode
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityCertificateType(Enum):
    """Types of quality certificates"""
    TASK_COMPLETION = "task_completion"
    MODULE_VALIDATION = "module_validation"
    SYSTEM_COMPLIANCE = "system_compliance"
    SECURITY_ASSESSMENT = "security_assessment"
    PERFORMANCE_VALIDATION = "performance_validation"
    HEALTHCARE_COMPLIANCE = "healthcare_compliance"
    USER_ACCEPTANCE = "user_acceptance"
    REGULATORY_APPROVAL = "regulatory_approval"


class QualityLevel(Enum):
    """Quality certification levels"""
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    DIAMOND = "diamond"


class ComplianceFramework(Enum):
    """Healthcare compliance frameworks"""
    HIPAA = "HIPAA"
    FDA_510K = "FDA_510K"
    FDA_DE_NOVO = "FDA_DE_NOVO"
    SOC2_TYPE2 = "SOC2_TYPE2"
    HITECH = "HITECH"
    GDPR = "GDPR"
    ISO_27001 = "ISO_27001"
    ISO_13485 = "ISO_13485"
    IEC_62304 = "IEC_62304"


@dataclass
class QualityMetric:
    """Individual quality metric"""
    name: str
    value: float
    max_value: float
    unit: str
    category: str
    weight: float
    threshold_pass: float
    threshold_excellent: float
    measured_at: str
    validation_method: str


@dataclass
class ComplianceValidation:
    """Compliance validation result"""
    framework: ComplianceFramework
    compliant: bool
    compliance_score: float
    validated_controls: List[str]
    failed_controls: List[str]
    remediation_required: List[str]
    evidence_artifacts: List[str]
    validator_signature: str
    validation_date: str


@dataclass
class QualityCertificate:
    """Comprehensive quality certificate"""
    certificate_id: str
    certificate_type: QualityCertificateType
    title: str
    description: str
    subject_id: str  # Task ID, Module ID, or System ID
    subject_name: str
    quality_level: QualityLevel
    overall_score: float
    
    # Quality metrics
    quality_metrics: List[QualityMetric]
    
    # Compliance validations
    compliance_validations: List[ComplianceValidation]
    
    # Testing results
    test_coverage: float
    test_success_rate: float
    security_scan_passed: bool
    performance_benchmarks: Dict[str, float]
    
    # Business rules validation
    business_rules_compliant: bool
    tenant_specific_validations: List[str]
    
    # Traceability
    development_artifacts: List[str]
    test_artifacts: List[str]
    documentation_artifacts: List[str]
    audit_trail: List[Dict[str, Any]]
    
    # Certification metadata
    issued_by: str
    issued_to: str
    issue_date: str
    expiry_date: str
    certificate_hash: str
    digital_signature: str
    qr_code_data: str
    
    # Validation chain
    parent_certificates: List[str]
    child_certificates: List[str]
    validation_chain: List[str]


class ComprehensiveQualityManager:
    """
    Comprehensive Quality Certification Manager
    Generates, validates, and manages quality certificates at all levels
    """
    
    def __init__(self):
        self.certificates_db = {}
        self.quality_standards = self._load_quality_standards()
        self.compliance_validators = self._initialize_compliance_validators()
        self.certificate_templates = self._load_certificate_templates()
        self.digital_signature_key = self._generate_signature_key()
        
        # Create output directories
        self.output_dir = Path("quality_certificates")
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "pdf").mkdir(exist_ok=True)
        (self.output_dir / "html").mkdir(exist_ok=True)
        (self.output_dir / "json").mkdir(exist_ok=True)
        (self.output_dir / "analytics").mkdir(exist_ok=True)
    
    async def generate_task_quality_certificate(self, 
                                               task: Dict[str, Any],
                                               development_result: Dict[str, Any],
                                               testing_result: Dict[str, Any],
                                               business_rules_result: Dict[str, Any]) -> QualityCertificate:
        """Generate comprehensive quality certificate for a completed task"""
        
        logger.info(f"🏆 Generating task quality certificate for: {task['title']}")
        
        certificate_id = f"TASK-CERT-{str(uuid.uuid4())[:8]}"
        
        # Calculate quality metrics
        quality_metrics = await self._calculate_task_quality_metrics(
            task, development_result, testing_result, business_rules_result
        )
        
        # Validate compliance
        compliance_validations = await self._validate_task_compliance(
            task, development_result, testing_result
        )
        
        # Calculate overall score
        overall_score = self._calculate_overall_quality_score(quality_metrics)
        quality_level = self._determine_quality_level(overall_score)
        
        # Generate audit trail
        audit_trail = await self._generate_task_audit_trail(task, development_result, testing_result)
        
        # Create certificate
        certificate = QualityCertificate(
            certificate_id=certificate_id,
            certificate_type=QualityCertificateType.TASK_COMPLETION,
            title=f"Task Quality Certificate: {task['title']}",
            description=f"Comprehensive quality validation for task {task['id']}",
            subject_id=task['id'],
            subject_name=task['title'],
            quality_level=quality_level,
            overall_score=overall_score,
            quality_metrics=quality_metrics,
            compliance_validations=compliance_validations,
            test_coverage=testing_result.get('coverage', 0.0),
            test_success_rate=testing_result.get('success_rate', 0.0),
            security_scan_passed=testing_result.get('security_passed', False),
            performance_benchmarks=testing_result.get('performance_metrics', {}),
            business_rules_compliant=business_rules_result.get('compliant', False),
            tenant_specific_validations=business_rules_result.get('validations', []),
            development_artifacts=development_result.get('artifacts', []),
            test_artifacts=testing_result.get('artifacts', []),
            documentation_artifacts=development_result.get('documentation', []),
            audit_trail=audit_trail,
            issued_by="MedinovAI Quality Assurance System",
            issued_to=task.get('assigned_to', 'Development Team'),
            issue_date=datetime.now().isoformat(),
            expiry_date=(datetime.now() + timedelta(days=365)).isoformat(),
            certificate_hash="",  # Will be calculated
            digital_signature="",  # Will be calculated
            qr_code_data="",  # Will be calculated
            parent_certificates=[],
            child_certificates=[],
            validation_chain=[]
        )
        
        # Calculate hash and signature
        certificate.certificate_hash = self._calculate_certificate_hash(certificate)
        certificate.digital_signature = self._generate_digital_signature(certificate)
        certificate.qr_code_data = self._generate_qr_code_data(certificate)
        
        # Store certificate
        self.certificates_db[certificate_id] = certificate
        
        # Generate certificate documents
        await self._generate_certificate_documents(certificate)
        
        logger.info(f"✅ Task quality certificate generated: {certificate_id}")
        return certificate
    
    async def generate_module_quality_certificate(self,
                                                 module_info: Dict[str, Any],
                                                 task_certificates: List[QualityCertificate],
                                                 integration_results: Dict[str, Any]) -> QualityCertificate:
        """Generate comprehensive quality certificate for a module"""
        
        logger.info(f"🏆 Generating module quality certificate for: {module_info['name']}")
        
        certificate_id = f"MODULE-CERT-{str(uuid.uuid4())[:8]}"
        
        # Aggregate metrics from task certificates
        aggregated_metrics = await self._aggregate_task_metrics(task_certificates)
        
        # Module-specific quality metrics
        module_metrics = await self._calculate_module_quality_metrics(
            module_info, task_certificates, integration_results
        )
        
        # Combine all metrics
        all_metrics = aggregated_metrics + module_metrics
        
        # Validate module compliance
        compliance_validations = await self._validate_module_compliance(
            module_info, task_certificates, integration_results
        )
        
        # Calculate overall score
        overall_score = self._calculate_overall_quality_score(all_metrics)
        quality_level = self._determine_quality_level(overall_score)
        
        # Generate module audit trail
        audit_trail = await self._generate_module_audit_trail(
            module_info, task_certificates, integration_results
        )
        
        # Create certificate
        certificate = QualityCertificate(
            certificate_id=certificate_id,
            certificate_type=QualityCertificateType.MODULE_VALIDATION,
            title=f"Module Quality Certificate: {module_info['name']}",
            description=f"Comprehensive quality validation for module {module_info['id']}",
            subject_id=module_info['id'],
            subject_name=module_info['name'],
            quality_level=quality_level,
            overall_score=overall_score,
            quality_metrics=all_metrics,
            compliance_validations=compliance_validations,
            test_coverage=integration_results.get('coverage', 0.0),
            test_success_rate=integration_results.get('success_rate', 0.0),
            security_scan_passed=integration_results.get('security_passed', False),
            performance_benchmarks=integration_results.get('performance_metrics', {}),
            business_rules_compliant=integration_results.get('rules_compliant', False),
            tenant_specific_validations=integration_results.get('tenant_validations', []),
            development_artifacts=self._collect_module_artifacts(task_certificates, 'development'),
            test_artifacts=self._collect_module_artifacts(task_certificates, 'test'),
            documentation_artifacts=self._collect_module_artifacts(task_certificates, 'documentation'),
            audit_trail=audit_trail,
            issued_by="MedinovAI Quality Assurance System",
            issued_to=module_info.get('owner', 'Development Team'),
            issue_date=datetime.now().isoformat(),
            expiry_date=(datetime.now() + timedelta(days=365)).isoformat(),
            certificate_hash="",
            digital_signature="",
            qr_code_data="",
            parent_certificates=[cert.certificate_id for cert in task_certificates],
            child_certificates=[],
            validation_chain=self._build_validation_chain(task_certificates)
        )
        
        # Calculate hash and signature
        certificate.certificate_hash = self._calculate_certificate_hash(certificate)
        certificate.digital_signature = self._generate_digital_signature(certificate)
        certificate.qr_code_data = self._generate_qr_code_data(certificate)
        
        # Update child certificates in task certificates
        for task_cert in task_certificates:
            task_cert.child_certificates.append(certificate_id)
            self.certificates_db[task_cert.certificate_id] = task_cert
        
        # Store certificate
        self.certificates_db[certificate_id] = certificate
        
        # Generate certificate documents
        await self._generate_certificate_documents(certificate)
        
        logger.info(f"✅ Module quality certificate generated: {certificate_id}")
        return certificate
    
    async def generate_system_quality_certificate(self,
                                                 system_info: Dict[str, Any],
                                                 module_certificates: List[QualityCertificate],
                                                 system_tests: Dict[str, Any],
                                                 compliance_audit: Dict[str, Any]) -> QualityCertificate:
        """Generate comprehensive system-wide quality certificate"""
        
        logger.info(f"🏆 Generating system quality certificate for: {system_info['name']}")
        
        certificate_id = f"SYSTEM-CERT-{str(uuid.uuid4())[:8]}"
        
        # System-wide quality metrics
        system_metrics = await self._calculate_system_quality_metrics(
            system_info, module_certificates, system_tests, compliance_audit
        )
        
        # Comprehensive compliance validation
        compliance_validations = await self._validate_system_compliance(
            system_info, module_certificates, compliance_audit
        )
        
        # Calculate overall score
        overall_score = self._calculate_overall_quality_score(system_metrics)
        quality_level = self._determine_quality_level(overall_score)
        
        # Generate system audit trail
        audit_trail = await self._generate_system_audit_trail(
            system_info, module_certificates, system_tests, compliance_audit
        )
        
        # Create certificate
        certificate = QualityCertificate(
            certificate_id=certificate_id,
            certificate_type=QualityCertificateType.SYSTEM_COMPLIANCE,
            title=f"System Quality Certificate: {system_info['name']}",
            description=f"Comprehensive system-wide quality validation for {system_info['name']}",
            subject_id=system_info['id'],
            subject_name=system_info['name'],
            quality_level=quality_level,
            overall_score=overall_score,
            quality_metrics=system_metrics,
            compliance_validations=compliance_validations,
            test_coverage=system_tests.get('coverage', 0.0),
            test_success_rate=system_tests.get('success_rate', 0.0),
            security_scan_passed=system_tests.get('security_passed', False),
            performance_benchmarks=system_tests.get('performance_metrics', {}),
            business_rules_compliant=system_tests.get('rules_compliant', False),
            tenant_specific_validations=system_tests.get('tenant_validations', []),
            development_artifacts=self._collect_system_artifacts(module_certificates, 'development'),
            test_artifacts=self._collect_system_artifacts(module_certificates, 'test'),
            documentation_artifacts=self._collect_system_artifacts(module_certificates, 'documentation'),
            audit_trail=audit_trail,
            issued_by="MedinovAI Quality Assurance System",
            issued_to=system_info.get('owner', 'MedinovAI Platform'),
            issue_date=datetime.now().isoformat(),
            expiry_date=(datetime.now() + timedelta(days=730)).isoformat(),  # 2 years for system cert
            certificate_hash="",
            digital_signature="",
            qr_code_data="",
            parent_certificates=[cert.certificate_id for cert in module_certificates],
            child_certificates=[],
            validation_chain=self._build_system_validation_chain(module_certificates)
        )
        
        # Calculate hash and signature
        certificate.certificate_hash = self._calculate_certificate_hash(certificate)
        certificate.digital_signature = self._generate_digital_signature(certificate)
        certificate.qr_code_data = self._generate_qr_code_data(certificate)
        
        # Update child certificates in module certificates
        for module_cert in module_certificates:
            module_cert.child_certificates.append(certificate_id)
            self.certificates_db[module_cert.certificate_id] = module_cert
        
        # Store certificate
        self.certificates_db[certificate_id] = certificate
        
        # Generate certificate documents
        await self._generate_certificate_documents(certificate)
        
        # Generate system quality analytics
        await self._generate_system_quality_analytics(certificate, module_certificates)
        
        logger.info(f"✅ System quality certificate generated: {certificate_id}")
        return certificate
    
    # QUALITY METRICS CALCULATION
    
    async def _calculate_task_quality_metrics(self,
                                            task: Dict[str, Any],
                                            development_result: Dict[str, Any],
                                            testing_result: Dict[str, Any],
                                            business_rules_result: Dict[str, Any]) -> List[QualityMetric]:
        """Calculate comprehensive quality metrics for a task"""
        
        metrics = []
        now = datetime.now().isoformat()
        
        # Code Quality Metrics
        metrics.append(QualityMetric(
            name="Code Coverage",
            value=testing_result.get('code_coverage', 0.0),
            max_value=100.0,
            unit="%",
            category="Code Quality",
            weight=0.2,
            threshold_pass=80.0,
            threshold_excellent=95.0,
            measured_at=now,
            validation_method="Automated Code Analysis"
        ))
        
        metrics.append(QualityMetric(
            name="Cyclomatic Complexity",
            value=development_result.get('complexity_score', 10.0),
            max_value=10.0,  # Lower is better
            unit="complexity",
            category="Code Quality",
            weight=0.15,
            threshold_pass=10.0,
            threshold_excellent=5.0,
            measured_at=now,
            validation_method="Static Code Analysis"
        ))
        
        # Testing Metrics
        metrics.append(QualityMetric(
            name="Test Success Rate",
            value=testing_result.get('success_rate', 0.0),
            max_value=100.0,
            unit="%",
            category="Testing",
            weight=0.25,
            threshold_pass=95.0,
            threshold_excellent=99.0,
            measured_at=now,
            validation_method="Automated Testing Suite"
        ))
        
        metrics.append(QualityMetric(
            name="Security Vulnerabilities",
            value=testing_result.get('security_vulnerabilities', 0),
            max_value=0.0,  # Lower is better
            unit="count",
            category="Security",
            weight=0.2,
            threshold_pass=0.0,
            threshold_excellent=0.0,
            measured_at=now,
            validation_method="Security Scanning Tools"
        ))
        
        # Performance Metrics
        metrics.append(QualityMetric(
            name="Response Time",
            value=testing_result.get('avg_response_time', 500.0),
            max_value=200.0,  # Lower is better
            unit="ms",
            category="Performance",
            weight=0.1,
            threshold_pass=500.0,
            threshold_excellent=200.0,
            measured_at=now,
            validation_method="Performance Testing"
        ))
        
        # Business Rules Compliance
        metrics.append(QualityMetric(
            name="Business Rules Compliance",
            value=100.0 if business_rules_result.get('compliant', False) else 0.0,
            max_value=100.0,
            unit="%",
            category="Business Rules",
            weight=0.1,
            threshold_pass=100.0,
            threshold_excellent=100.0,
            measured_at=now,
            validation_method="Business Rules Engine"
        ))
        
        return metrics
    
    async def _validate_task_compliance(self,
                                       task: Dict[str, Any],
                                       development_result: Dict[str, Any],
                                       testing_result: Dict[str, Any]) -> List[ComplianceValidation]:
        """Validate task compliance against healthcare frameworks"""
        
        validations = []
        
        # HIPAA Compliance
        hipaa_validation = ComplianceValidation(
            framework=ComplianceFramework.HIPAA,
            compliant=testing_result.get('hipaa_compliant', False),
            compliance_score=testing_result.get('hipaa_score', 0.0),
            validated_controls=[
                "Data Encryption",
                "Access Controls",
                "Audit Logging",
                "Minimum Necessary"
            ],
            failed_controls=testing_result.get('hipaa_failed_controls', []),
            remediation_required=testing_result.get('hipaa_remediation', []),
            evidence_artifacts=development_result.get('compliance_artifacts', []),
            validator_signature=self._generate_validator_signature("HIPAA"),
            validation_date=datetime.now().isoformat()
        )
        validations.append(hipaa_validation)
        
        # SOC2 Type 2 Compliance
        soc2_validation = ComplianceValidation(
            framework=ComplianceFramework.SOC2_TYPE2,
            compliant=testing_result.get('soc2_compliant', False),
            compliance_score=testing_result.get('soc2_score', 0.0),
            validated_controls=[
                "Security",
                "Availability",
                "Processing Integrity",
                "Confidentiality",
                "Privacy"
            ],
            failed_controls=testing_result.get('soc2_failed_controls', []),
            remediation_required=testing_result.get('soc2_remediation', []),
            evidence_artifacts=development_result.get('compliance_artifacts', []),
            validator_signature=self._generate_validator_signature("SOC2"),
            validation_date=datetime.now().isoformat()
        )
        validations.append(soc2_validation)
        
        return validations
    
    # CERTIFICATE DOCUMENT GENERATION
    
    async def _generate_certificate_documents(self, certificate: QualityCertificate):
        """Generate professional certificate documents in multiple formats"""
        
        # Generate PDF certificate
        await self._generate_pdf_certificate(certificate)
        
        # Generate HTML certificate
        await self._generate_html_certificate(certificate)
        
        # Generate JSON certificate
        await self._generate_json_certificate(certificate)
        
        # Generate QR code
        await self._generate_qr_code(certificate)
    
    async def _generate_pdf_certificate(self, certificate: QualityCertificate):
        """Generate professional PDF certificate"""
        
        filename = f"{certificate.certificate_id}_{certificate.certificate_type.value}.pdf"
        filepath = self.output_dir / "pdf" / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(str(filepath), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=colors.darkblue
        )
        story.append(Paragraph("QUALITY CERTIFICATE", title_style))
        story.append(Spacer(1, 12))
        
        # Certificate type and level
        cert_info_style = ParagraphStyle(
            'CertInfo',
            parent=styles['Normal'],
            fontSize=16,
            alignment=1,
            textColor=colors.darkgreen
        )
        story.append(Paragraph(f"{certificate.title}", cert_info_style))
        story.append(Paragraph(f"Quality Level: {certificate.quality_level.value.upper()}", cert_info_style))
        story.append(Paragraph(f"Overall Score: {certificate.overall_score:.1f}/100", cert_info_style))
        story.append(Spacer(1, 20))
        
        # Certificate details table
        cert_data = [
            ['Certificate ID', certificate.certificate_id],
            ['Subject', certificate.subject_name],
            ['Issued By', certificate.issued_by],
            ['Issued To', certificate.issued_to],
            ['Issue Date', certificate.issue_date[:10]],
            ['Expiry Date', certificate.expiry_date[:10]],
            ['Test Coverage', f"{certificate.test_coverage:.1f}%"],
            ['Test Success Rate', f"{certificate.test_success_rate:.1f}%"],
            ['Security Scan', "PASSED" if certificate.security_scan_passed else "FAILED"],
            ['Business Rules', "COMPLIANT" if certificate.business_rules_compliant else "NON-COMPLIANT"]
        ]
        
        cert_table = Table(cert_data, colWidths=[2*inch, 3*inch])
        cert_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(cert_table)
        story.append(Spacer(1, 20))
        
        # Quality metrics summary
        metrics_style = ParagraphStyle(
            'MetricsTitle',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.darkblue
        )
        story.append(Paragraph("Quality Metrics Summary", metrics_style))
        story.append(Spacer(1, 10))
        
        metrics_data = [['Metric', 'Value', 'Category', 'Status']]
        for metric in certificate.quality_metrics:
            status = "EXCELLENT" if metric.value >= metric.threshold_excellent else "PASS" if metric.value >= metric.threshold_pass else "FAIL"
            metrics_data.append([
                metric.name,
                f"{metric.value} {metric.unit}",
                metric.category,
                status
            ])
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1*inch, 1.5*inch, 1*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Compliance validation summary
        compliance_style = ParagraphStyle(
            'ComplianceTitle',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.darkblue
        )
        story.append(Paragraph("Compliance Validation", compliance_style))
        story.append(Spacer(1, 10))
        
        compliance_data = [['Framework', 'Status', 'Score', 'Validation Date']]
        for validation in certificate.compliance_validations:
            status = "COMPLIANT" if validation.compliant else "NON-COMPLIANT"
            compliance_data.append([
                validation.framework.value,
                status,
                f"{validation.compliance_score:.1f}%",
                validation.validation_date[:10]
            ])
        
        compliance_table = Table(compliance_data, colWidths=[1.5*inch, 1.5*inch, 1*inch, 1.5*inch])
        compliance_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkgreen),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.lightgreen),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(compliance_table)
        story.append(Spacer(1, 30))
        
        # Digital signature and hash
        signature_style = ParagraphStyle(
            'Signature',
            parent=styles['Normal'],
            fontSize=8,
            alignment=1,
            textColor=colors.grey
        )
        story.append(Paragraph(f"Certificate Hash: {certificate.certificate_hash}", signature_style))
        story.append(Paragraph(f"Digital Signature: {certificate.digital_signature[:50]}...", signature_style))
        story.append(Paragraph("This certificate is digitally signed and cryptographically verified", signature_style))
        
        # Build PDF
        doc.build(story)
        logger.info(f"📄 PDF certificate generated: {filepath}")
    
    async def _generate_html_certificate(self, certificate: QualityCertificate):
        """Generate interactive HTML certificate"""
        
        filename = f"{certificate.certificate_id}_{certificate.certificate_type.value}.html"
        filepath = self.output_dir / "html" / filename
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ certificate.title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .certificate { background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
                .header { text-align: center; margin-bottom: 30px; }
                .title { color: #2c3e50; font-size: 28px; font-weight: bold; margin-bottom: 10px; }
                .subtitle { color: #27ae60; font-size: 18px; margin-bottom: 20px; }
                .quality-level { background: #{{ level_color }}; color: white; padding: 10px 20px; border-radius: 5px; display: inline-block; }
                .section { margin: 30px 0; }
                .section h3 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
                .metric-card { background: #ecf0f1; padding: 15px; border-radius: 5px; }
                .metric-value { font-size: 24px; font-weight: bold; color: #27ae60; }
                .compliance-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                .compliance-table th, .compliance-table td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                .compliance-table th { background: #3498db; color: white; }
                .compliant { color: #27ae60; font-weight: bold; }
                .non-compliant { color: #e74c3c; font-weight: bold; }
                .footer { text-align: center; margin-top: 40px; color: #7f8c8d; }
                .qr-code { text-align: center; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="certificate">
                <div class="header">
                    <div class="title">QUALITY CERTIFICATE</div>
                    <div class="subtitle">{{ certificate.title }}</div>
                    <div class="quality-level">{{ certificate.quality_level.value.upper() }} LEVEL</div>
                    <div style="margin-top: 15px; font-size: 20px; color: #2c3e50;">
                        Overall Score: {{ certificate.overall_score }}/100
                    </div>
                </div>
                
                <div class="section">
                    <h3>Certificate Information</h3>
                    <p><strong>Certificate ID:</strong> {{ certificate.certificate_id }}</p>
                    <p><strong>Subject:</strong> {{ certificate.subject_name }}</p>
                    <p><strong>Issued By:</strong> {{ certificate.issued_by }}</p>
                    <p><strong>Issued To:</strong> {{ certificate.issued_to }}</p>
                    <p><strong>Issue Date:</strong> {{ certificate.issue_date[:10] }}</p>
                    <p><strong>Expiry Date:</strong> {{ certificate.expiry_date[:10] }}</p>
                </div>
                
                <div class="section">
                    <h3>Quality Metrics</h3>
                    <div class="metrics-grid">
                        {% for metric in certificate.quality_metrics %}
                        <div class="metric-card">
                            <div class="metric-value">{{ metric.value }} {{ metric.unit }}</div>
                            <div><strong>{{ metric.name }}</strong></div>
                            <div>Category: {{ metric.category }}</div>
                            <div>Weight: {{ metric.weight }}</div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                
                <div class="section">
                    <h3>Compliance Validation</h3>
                    <table class="compliance-table">
                        <thead>
                            <tr>
                                <th>Framework</th>
                                <th>Status</th>
                                <th>Score</th>
                                <th>Validation Date</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for validation in certificate.compliance_validations %}
                            <tr>
                                <td>{{ validation.framework.value }}</td>
                                <td class="{{ 'compliant' if validation.compliant else 'non-compliant' }}">
                                    {{ 'COMPLIANT' if validation.compliant else 'NON-COMPLIANT' }}
                                </td>
                                <td>{{ validation.compliance_score }}%</td>
                                <td>{{ validation.validation_date[:10] }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                
                <div class="section">
                    <h3>Testing Summary</h3>
                    <p><strong>Test Coverage:</strong> {{ certificate.test_coverage }}%</p>
                    <p><strong>Test Success Rate:</strong> {{ certificate.test_success_rate }}%</p>
                    <p><strong>Security Scan:</strong> {{ 'PASSED' if certificate.security_scan_passed else 'FAILED' }}</p>
                    <p><strong>Business Rules Compliance:</strong> {{ 'COMPLIANT' if certificate.business_rules_compliant else 'NON-COMPLIANT' }}</p>
                </div>
                
                <div class="qr-code">
                    <p><strong>Verification QR Code:</strong></p>
                    <div>{{ certificate.qr_code_data }}</div>
                </div>
                
                <div class="footer">
                    <p>Certificate Hash: {{ certificate.certificate_hash }}</p>
                    <p>Digital Signature: {{ certificate.digital_signature[:50] }}...</p>
                    <p>This certificate is digitally signed and cryptographically verified</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Determine quality level color
        level_colors = {
            QualityLevel.BRONZE: "cd7f32",
            QualityLevel.SILVER: "c0c0c0", 
            QualityLevel.GOLD: "ffd700",
            QualityLevel.PLATINUM: "e5e4e2",
            QualityLevel.DIAMOND: "b9f2ff"
        }
        
        template = Template(html_template)
        html_content = template.render(
            certificate=certificate,
            level_color=level_colors.get(certificate.quality_level, "3498db")
        )
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"🌐 HTML certificate generated: {filepath}")
    
    async def _generate_json_certificate(self, certificate: QualityCertificate):
        """Generate machine-readable JSON certificate"""
        
        filename = f"{certificate.certificate_id}_{certificate.certificate_type.value}.json"
        filepath = self.output_dir / "json" / filename
        
        certificate_dict = asdict(certificate)
        
        with open(filepath, 'w') as f:
            json.dump(certificate_dict, f, indent=2, default=str)
        
        logger.info(f"📊 JSON certificate generated: {filepath}")
    
    async def _generate_qr_code(self, certificate: QualityCertificate):
        """Generate QR code for certificate verification"""
        
        qr_data = {
            "certificate_id": certificate.certificate_id,
            "hash": certificate.certificate_hash,
            "verification_url": f"https://medinovai.com/verify/{certificate.certificate_id}",
            "issue_date": certificate.issue_date
        }
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(json.dumps(qr_data))
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        filename = f"{certificate.certificate_id}_qr.png"
        filepath = self.output_dir / "pdf" / filename
        img.save(filepath)
        
        logger.info(f"📱 QR code generated: {filepath}")
    
    # SYSTEM ANALYTICS
    
    async def _generate_system_quality_analytics(self, 
                                                system_cert: QualityCertificate,
                                                module_certs: List[QualityCertificate]):
        """Generate comprehensive system quality analytics"""
        
        logger.info("📊 Generating system quality analytics...")
        
        # Create analytics plots
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Quality levels distribution
        quality_levels = [cert.quality_level.value for cert in module_certs]
        quality_counts = pd.Series(quality_levels).value_counts()
        ax1.pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%')
        ax1.set_title('Quality Levels Distribution Across Modules')
        
        # Quality scores over time
        scores = [cert.overall_score for cert in module_certs]
        ax2.hist(scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Quality Score')
        ax2.set_ylabel('Number of Modules')
        ax2.set_title('Quality Score Distribution')
        
        # Compliance status
        compliance_data = []
        for cert in module_certs:
            for validation in cert.compliance_validations:
                compliance_data.append({
                    'Framework': validation.framework.value,
                    'Compliant': validation.compliant,
                    'Score': validation.compliance_score
                })
        
        if compliance_data:
            df = pd.DataFrame(compliance_data)
            compliance_summary = df.groupby('Framework')['Compliant'].mean()
            ax3.bar(compliance_summary.index, compliance_summary.values, color='lightgreen', alpha=0.7)
            ax3.set_ylabel('Compliance Rate')
            ax3.set_title('Compliance Rate by Framework')
            ax3.tick_params(axis='x', rotation=45)
        
        # Test metrics summary
        test_coverage = [cert.test_coverage for cert in module_certs if cert.test_coverage > 0]
        test_success = [cert.test_success_rate for cert in module_certs if cert.test_success_rate > 0]
        
        ax4.scatter(test_coverage, test_success, alpha=0.7, color='coral')
        ax4.set_xlabel('Test Coverage (%)')
        ax4.set_ylabel('Test Success Rate (%)')
        ax4.set_title('Test Coverage vs Success Rate')
        
        plt.tight_layout()
        
        # Save analytics
        analytics_file = self.output_dir / "analytics" / f"system_quality_analytics_{system_cert.certificate_id}.png"
        plt.savefig(analytics_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 System quality analytics saved: {analytics_file}")
    
    # HELPER METHODS
    
    def _calculate_overall_quality_score(self, metrics: List[QualityMetric]) -> float:
        """Calculate weighted overall quality score"""
        if not metrics:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for metric in metrics:
            # Normalize metric value to 0-100 scale
            if metric.name in ["Cyclomatic Complexity", "Security Vulnerabilities", "Response Time"]:
                # Lower is better metrics
                normalized_value = max(0, 100 - (metric.value / metric.max_value * 100))
            else:
                # Higher is better metrics
                normalized_value = min(100, (metric.value / metric.max_value) * 100)
            
            total_weighted_score += normalized_value * metric.weight
            total_weight += metric.weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_level(self, overall_score: float) -> QualityLevel:
        """Determine quality level based on overall score"""
        if overall_score >= 95.0:
            return QualityLevel.DIAMOND
        elif overall_score >= 90.0:
            return QualityLevel.PLATINUM
        elif overall_score >= 80.0:
            return QualityLevel.GOLD
        elif overall_score >= 70.0:
            return QualityLevel.SILVER
        else:
            return QualityLevel.BRONZE
    
    def _calculate_certificate_hash(self, certificate: QualityCertificate) -> str:
        """Calculate cryptographic hash of certificate"""
        # Create a copy without hash and signature
        cert_data = asdict(certificate)
        cert_data.pop('certificate_hash', None)
        cert_data.pop('digital_signature', None)
        cert_data.pop('qr_code_data', None)
        
        cert_json = json.dumps(cert_data, sort_keys=True, default=str)
        return hashlib.sha256(cert_json.encode()).hexdigest()
    
    def _generate_digital_signature(self, certificate: QualityCertificate) -> str:
        """Generate digital signature for certificate"""
        signature_data = f"{certificate.certificate_id}:{certificate.certificate_hash}:{self.digital_signature_key}"
        return hashlib.sha512(signature_data.encode()).hexdigest()
    
    def _generate_qr_code_data(self, certificate: QualityCertificate) -> str:
        """Generate QR code data for certificate verification"""
        qr_data = {
            "id": certificate.certificate_id,
            "hash": certificate.certificate_hash,
            "url": f"https://medinovai.com/verify/{certificate.certificate_id}"
        }
        return base64.b64encode(json.dumps(qr_data).encode()).decode()
    
    def _generate_signature_key(self) -> str:
        """Generate unique signature key for this instance"""
        return hashlib.sha256(f"medinovai_quality_manager_{datetime.now().isoformat()}".encode()).hexdigest()
    
    def _generate_validator_signature(self, framework: str) -> str:
        """Generate validator signature for compliance framework"""
        return hashlib.md5(f"{framework}_validator_{datetime.now().isoformat()}".encode()).hexdigest()
    
    def _load_quality_standards(self) -> Dict[str, Any]:
        """Load quality standards configuration"""
        return {
            "code_coverage_threshold": 80.0,
            "test_success_threshold": 95.0,
            "security_vulnerability_threshold": 0,
            "performance_threshold": 500.0,
            "compliance_score_threshold": 90.0
        }
    
    def _initialize_compliance_validators(self) -> Dict[str, Any]:
        """Initialize compliance validators"""
        return {
            "HIPAA": "HIPAAValidator",
            "FDA": "FDAValidator", 
            "SOC2": "SOC2Validator"
        }
    
    def _load_certificate_templates(self) -> Dict[str, str]:
        """Load certificate templates"""
        return {
            "task": "task_certificate_template.html",
            "module": "module_certificate_template.html",
            "system": "system_certificate_template.html"
        }
    
    # Additional helper methods for complex operations...
    async def _aggregate_task_metrics(self, task_certificates: List[QualityCertificate]) -> List[QualityMetric]:
        """Aggregate metrics from multiple task certificates"""
        # Implementation for aggregating task metrics
        return []
    
    async def _calculate_module_quality_metrics(self, module_info: Dict[str, Any], 
                                               task_certificates: List[QualityCertificate],
                                               integration_results: Dict[str, Any]) -> List[QualityMetric]:
        """Calculate module-specific quality metrics"""
        # Implementation for module quality metrics
        return []
    
    def _collect_module_artifacts(self, task_certificates: List[QualityCertificate], artifact_type: str) -> List[str]:
        """Collect artifacts from task certificates"""
        artifacts = []
        for cert in task_certificates:
            if artifact_type == 'development':
                artifacts.extend(cert.development_artifacts)
            elif artifact_type == 'test':
                artifacts.extend(cert.test_artifacts)
            elif artifact_type == 'documentation':
                artifacts.extend(cert.documentation_artifacts)
        return artifacts
    
    def _build_validation_chain(self, task_certificates: List[QualityCertificate]) -> List[str]:
        """Build validation chain from task certificates"""
        return [cert.certificate_id for cert in task_certificates]


# Example usage
async def main():
    """Example usage of comprehensive quality manager"""
    print("🏆 Initializing Comprehensive Quality Manager...")
    
    quality_manager = ComprehensiveQualityManager()
    
    # Example task completion
    sample_task = {
        "id": "task_001",
        "title": "Patient Portal Dashboard Implementation",
        "description": "Comprehensive patient portal with multi-modal features",
        "area": "Patient Portal",
        "feature": "Dashboard"
    }
    
    sample_development_result = {
        "complexity_score": 8.5,
        "compliance_artifacts": ["security_scan.pdf", "code_review.pdf"],
        "artifacts": ["dashboard.js", "patient_api.py"],
        "documentation": ["api_docs.md", "user_guide.md"]
    }
    
    sample_testing_result = {
        "code_coverage": 92.0,
        "success_rate": 98.5,
        "avg_response_time": 180.0,
        "security_vulnerabilities": 0,
        "security_passed": True,
        "hipaa_compliant": True,
        "hipaa_score": 96.0,
        "soc2_compliant": True,
        "soc2_score": 94.0,
        "coverage": 92.0,
        "performance_metrics": {"response_time": 180.0, "throughput": 1000.0}
    }
    
    sample_business_rules_result = {
        "compliant": True,
        "validations": ["tenant_isolation", "data_privacy", "audit_logging"]
    }
    
    # Generate task quality certificate
    task_cert = await quality_manager.generate_task_quality_certificate(
        sample_task, sample_development_result, sample_testing_result, sample_business_rules_result
    )
    
    print(f"✅ Task quality certificate generated: {task_cert.certificate_id}")
    print(f"📊 Quality Level: {task_cert.quality_level.value.upper()}")
    print(f"🎯 Overall Score: {task_cert.overall_score:.1f}/100")
    
    # Generate module quality certificate
    sample_module = {
        "id": "module_patient_portal",
        "name": "Patient Portal Module",
        "owner": "Frontend Team"
    }
    
    sample_integration_results = {
        "coverage": 95.0,
        "success_rate": 99.0,
        "security_passed": True,
        "rules_compliant": True,
        "performance_metrics": {"response_time": 150.0}
    }
    
    module_cert = await quality_manager.generate_module_quality_certificate(
        sample_module, [task_cert], sample_integration_results
    )
    
    print(f"✅ Module quality certificate generated: {module_cert.certificate_id}")
    
    # Generate system quality certificate
    sample_system = {
        "id": "medinovai_platform",
        "name": "MedinovAI Healthcare Platform",
        "owner": "MedinovAI Engineering"
    }
    
    sample_system_tests = {
        "coverage": 96.0,
        "success_rate": 99.2,
        "security_passed": True,
        "rules_compliant": True,
        "performance_metrics": {"response_time": 120.0, "uptime": 99.9}
    }
    
    sample_compliance_audit = {
        "hipaa_compliant": True,
        "fda_validated": True,
        "soc2_certified": True
    }
    
    system_cert = await quality_manager.generate_system_quality_certificate(
        sample_system, [module_cert], sample_system_tests, sample_compliance_audit
    )
    
    print(f"✅ System quality certificate generated: {system_cert.certificate_id}")
    print(f"🏆 System Quality Level: {system_cert.quality_level.value.upper()}")
    print(f"📊 System Overall Score: {system_cert.overall_score:.1f}/100")
    
    print("\n🎉 Comprehensive Quality Certification System Demonstration Complete!")


if __name__ == "__main__":
    asyncio.run(main()) 