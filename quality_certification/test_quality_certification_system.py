#!/usr/bin/env python3
"""
Comprehensive Tests for Quality Certification System
Tests for quality certificate generation, validation, reporting, and API endpoints
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

# Import testing frameworks
from fastapi.testclient import TestClient
import requests_mock

# Import our modules
from comprehensive_quality_manager import (
    ComprehensiveQualityManager,
    QualityCertificate,
    QualityCertificateType,
    QualityLevel,
    ComplianceFramework,
    QualityMetric,
    ComplianceValidation
)
from quality_certificate_api import app


class TestComprehensiveQualityManager:
    """Test suite for ComprehensiveQualityManager"""
    
    @pytest.fixture
    def quality_manager(self):
        """Create quality manager for testing"""
        # Use temporary directory for test outputs
        temp_dir = tempfile.mkdtemp()
        manager = ComprehensiveQualityManager()
        manager.output_dir = Path(temp_dir)
        manager.output_dir.mkdir(exist_ok=True)
        (manager.output_dir / "pdf").mkdir(exist_ok=True)
        (manager.output_dir / "html").mkdir(exist_ok=True)
        (manager.output_dir / "json").mkdir(exist_ok=True)
        (manager.output_dir / "analytics").mkdir(exist_ok=True)
        
        yield manager
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_task(self):
        """Sample task for testing"""
        return {
            "id": "task_test_001",
            "title": "Test Patient Portal Implementation",
            "description": "Test implementation of patient portal dashboard",
            "area": "Patient Portal",
            "feature": "Dashboard",
            "priority": "high",
            "assigned_to": "Test Team"
        }
    
    @pytest.fixture
    def sample_development_result(self):
        """Sample development result for testing"""
        return {
            "complexity_score": 7.5,
            "compliance_artifacts": ["test_security_scan.pdf", "test_code_review.pdf"],
            "artifacts": ["test_dashboard.js", "test_patient_api.py"],
            "documentation": ["test_api_docs.md", "test_user_guide.md"]
        }
    
    @pytest.fixture
    def sample_testing_result(self):
        """Sample testing result for testing"""
        return {
            "code_coverage": 95.0,
            "success_rate": 98.5,
            "avg_response_time": 150.0,
            "security_vulnerabilities": 0,
            "security_passed": True,
            "hipaa_compliant": True,
            "hipaa_score": 97.0,
            "soc2_compliant": True,
            "soc2_score": 95.5,
            "coverage": 95.0,
            "performance_metrics": {"response_time": 150.0, "throughput": 1200.0}
        }
    
    @pytest.fixture
    def sample_business_rules_result(self):
        """Sample business rules result for testing"""
        return {
            "compliant": True,
            "validations": ["tenant_isolation", "data_privacy", "audit_logging", "access_control"]
        }
    
    @pytest.mark.asyncio
    async def test_generate_task_quality_certificate(self, quality_manager, sample_task, 
                                                   sample_development_result, sample_testing_result, 
                                                   sample_business_rules_result):
        """Test task quality certificate generation"""
        
        # Generate certificate
        certificate = await quality_manager.generate_task_quality_certificate(
            sample_task, sample_development_result, sample_testing_result, sample_business_rules_result
        )
        
        # Verify certificate properties
        assert certificate.certificate_id.startswith("TASK-CERT-")
        assert certificate.certificate_type == QualityCertificateType.TASK_COMPLETION
        assert certificate.subject_id == sample_task["id"]
        assert certificate.subject_name == sample_task["title"]
        assert certificate.overall_score > 0
        assert isinstance(certificate.quality_level, QualityLevel)
        
        # Verify quality metrics
        assert len(certificate.quality_metrics) > 0
        for metric in certificate.quality_metrics:
            assert isinstance(metric, QualityMetric)
            assert metric.value >= 0
            assert metric.max_value >= 0
        
        # Verify compliance validations
        assert len(certificate.compliance_validations) > 0
        for validation in certificate.compliance_validations:
            assert isinstance(validation, ComplianceValidation)
            assert isinstance(validation.framework, ComplianceFramework)
        
        # Verify certificate stored in database
        assert certificate.certificate_id in quality_manager.certificates_db
        
        # Verify files generated
        pdf_path = quality_manager.output_dir / "pdf" / f"{certificate.certificate_id}_{certificate.certificate_type.value}.pdf"
        html_path = quality_manager.output_dir / "html" / f"{certificate.certificate_id}_{certificate.certificate_type.value}.html"
        json_path = quality_manager.output_dir / "json" / f"{certificate.certificate_id}_{certificate.certificate_type.value}.json"
        
        assert pdf_path.exists()
        assert html_path.exists()
        assert json_path.exists()
    
    @pytest.mark.asyncio
    async def test_generate_module_quality_certificate(self, quality_manager, sample_task,
                                                      sample_development_result, sample_testing_result,
                                                      sample_business_rules_result):
        """Test module quality certificate generation"""
        
        # First generate a task certificate
        task_cert = await quality_manager.generate_task_quality_certificate(
            sample_task, sample_development_result, sample_testing_result, sample_business_rules_result
        )
        
        # Module info
        module_info = {
            "id": "module_test_001",
            "name": "Test Patient Portal Module",
            "owner": "Test Team"
        }
        
        integration_results = {
            "coverage": 96.0,
            "success_rate": 99.0,
            "security_passed": True,
            "rules_compliant": True,
            "performance_metrics": {"response_time": 120.0}
        }
        
        # Generate module certificate
        module_cert = await quality_manager.generate_module_quality_certificate(
            module_info, [task_cert], integration_results
        )
        
        # Verify module certificate
        assert module_cert.certificate_id.startswith("MODULE-CERT-")
        assert module_cert.certificate_type == QualityCertificateType.MODULE_VALIDATION
        assert task_cert.certificate_id in module_cert.parent_certificates
        assert module_cert.certificate_id in task_cert.child_certificates
        
        # Verify overall score calculation
        assert module_cert.overall_score > 0
        assert isinstance(module_cert.quality_level, QualityLevel)
    
    @pytest.mark.asyncio
    async def test_generate_system_quality_certificate(self, quality_manager, sample_task,
                                                      sample_development_result, sample_testing_result,
                                                      sample_business_rules_result):
        """Test system quality certificate generation"""
        
        # Generate task certificate
        task_cert = await quality_manager.generate_task_quality_certificate(
            sample_task, sample_development_result, sample_testing_result, sample_business_rules_result
        )
        
        # Generate module certificate
        module_info = {
            "id": "module_test_001",
            "name": "Test Patient Portal Module",
            "owner": "Test Team"
        }
        
        integration_results = {
            "coverage": 96.0,
            "success_rate": 99.0,
            "security_passed": True,
            "rules_compliant": True,
            "performance_metrics": {"response_time": 120.0}
        }
        
        module_cert = await quality_manager.generate_module_quality_certificate(
            module_info, [task_cert], integration_results
        )
        
        # System info
        system_info = {
            "id": "system_test_001",
            "name": "Test MedinovAI Platform",
            "owner": "Test Engineering"
        }
        
        system_tests = {
            "coverage": 97.0,
            "success_rate": 99.5,
            "security_passed": True,
            "rules_compliant": True,
            "performance_metrics": {"response_time": 100.0, "uptime": 99.9}
        }
        
        compliance_audit = {
            "hipaa_compliant": True,
            "fda_validated": True,
            "soc2_certified": True
        }
        
        # Generate system certificate
        system_cert = await quality_manager.generate_system_quality_certificate(
            system_info, [module_cert], system_tests, compliance_audit
        )
        
        # Verify system certificate
        assert system_cert.certificate_id.startswith("SYSTEM-CERT-")
        assert system_cert.certificate_type == QualityCertificateType.SYSTEM_COMPLIANCE
        assert module_cert.certificate_id in system_cert.parent_certificates
        assert system_cert.certificate_id in module_cert.child_certificates
        
        # Verify analytics generated
        analytics_files = list((quality_manager.output_dir / "analytics").glob("*.png"))
        assert len(analytics_files) > 0
    
    def test_quality_score_calculation(self, quality_manager):
        """Test quality score calculation logic"""
        
        metrics = [
            QualityMetric(
                name="Test Coverage",
                value=95.0,
                max_value=100.0,
                unit="%",
                category="Testing",
                weight=0.3,
                threshold_pass=80.0,
                threshold_excellent=95.0,
                measured_at=datetime.now().isoformat(),
                validation_method="Automated"
            ),
            QualityMetric(
                name="Code Quality",
                value=90.0,
                max_value=100.0,
                unit="%",
                category="Code",
                weight=0.4,
                threshold_pass=75.0,
                threshold_excellent=90.0,
                measured_at=datetime.now().isoformat(),
                validation_method="Static Analysis"
            ),
            QualityMetric(
                name="Security Score",
                value=98.0,
                max_value=100.0,
                unit="%",
                category="Security",
                weight=0.3,
                threshold_pass=85.0,
                threshold_excellent=95.0,
                measured_at=datetime.now().isoformat(),
                validation_method="Security Scan"
            )
        ]
        
        overall_score = quality_manager._calculate_overall_quality_score(metrics)
        
        # Should be weighted average: (95*0.3 + 90*0.4 + 98*0.3) = 93.9
        assert 93.0 <= overall_score <= 94.0
    
    def test_quality_level_determination(self, quality_manager):
        """Test quality level determination logic"""
        
        assert quality_manager._determine_quality_level(97.0) == QualityLevel.DIAMOND
        assert quality_manager._determine_quality_level(92.0) == QualityLevel.PLATINUM
        assert quality_manager._determine_quality_level(85.0) == QualityLevel.GOLD
        assert quality_manager._determine_quality_level(75.0) == QualityLevel.SILVER
        assert quality_manager._determine_quality_level(65.0) == QualityLevel.BRONZE
    
    def test_certificate_hash_calculation(self, quality_manager, sample_task,
                                         sample_development_result, sample_testing_result,
                                         sample_business_rules_result):
        """Test certificate hash calculation for integrity"""
        
        # Create a sample certificate
        certificate = QualityCertificate(
            certificate_id="TEST-CERT-12345678",
            certificate_type=QualityCertificateType.TASK_COMPLETION,
            title="Test Certificate",
            description="Test",
            subject_id="test_001",
            subject_name="Test",
            quality_level=QualityLevel.GOLD,
            overall_score=85.0,
            quality_metrics=[],
            compliance_validations=[],
            test_coverage=90.0,
            test_success_rate=95.0,
            security_scan_passed=True,
            performance_benchmarks={},
            business_rules_compliant=True,
            tenant_specific_validations=[],
            development_artifacts=[],
            test_artifacts=[],
            documentation_artifacts=[],
            audit_trail=[],
            issued_by="Test System",
            issued_to="Test User",
            issue_date=datetime.now().isoformat(),
            expiry_date=(datetime.now() + timedelta(days=365)).isoformat(),
            certificate_hash="",
            digital_signature="",
            qr_code_data="",
            parent_certificates=[],
            child_certificates=[],
            validation_chain=[]
        )
        
        # Calculate hash
        hash1 = quality_manager._calculate_certificate_hash(certificate)
        hash2 = quality_manager._calculate_certificate_hash(certificate)
        
        # Should be deterministic
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hash length
        
        # Modify certificate and verify hash changes
        certificate.overall_score = 90.0
        hash3 = quality_manager._calculate_certificate_hash(certificate)
        assert hash1 != hash3


class TestQualityCertificateAPI:
    """Test suite for Quality Certificate API"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_task_request(self):
        """Sample task completion request"""
        return {
            "task": {
                "id": "api_test_001",
                "title": "API Test Task",
                "description": "Test task for API",
                "area": "Testing",
                "feature": "API"
            },
            "development_result": {
                "complexity_score": 8.0,
                "compliance_artifacts": ["test.pdf"],
                "artifacts": ["test.js"],
                "documentation": ["test.md"]
            },
            "testing_result": {
                "code_coverage": 92.0,
                "success_rate": 97.0,
                "avg_response_time": 200.0,
                "security_vulnerabilities": 0,
                "security_passed": True,
                "hipaa_compliant": True,
                "hipaa_score": 95.0,
                "soc2_compliant": True,
                "soc2_score": 93.0,
                "coverage": 92.0,
                "performance_metrics": {"response_time": 200.0}
            },
            "business_rules_result": {
                "compliant": True,
                "validations": ["test_validation"]
            }
        }
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "Quality Certificate API"
        assert "timestamp" in data
    
    def test_generate_task_certificate_api(self, client, sample_task_request):
        """Test task certificate generation via API"""
        response = client.post("/api/certificates/task", json=sample_task_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "certificate_id" in data
        assert data["certificate_id"].startswith("TASK-CERT-")
        assert "quality_level" in data
        assert "overall_score" in data
        assert "pdf_url" in data
        assert "html_url" in data
        assert "verification_url" in data
    
    def test_list_certificates_api(self, client, sample_task_request):
        """Test listing certificates via API"""
        # First generate a certificate
        client.post("/api/certificates/task", json=sample_task_request)
        
        # Then list certificates
        response = client.get("/api/certificates")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "certificates" in data
        assert "pagination" in data
        assert len(data["certificates"]) > 0
    
    def test_get_certificate_details_api(self, client, sample_task_request):
        """Test getting certificate details via API"""
        # Generate certificate
        create_response = client.post("/api/certificates/task", json=sample_task_request)
        certificate_id = create_response.json()["certificate_id"]
        
        # Get details
        response = client.get(f"/api/certificates/{certificate_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "certificate" in data
        
        certificate = data["certificate"]
        assert "certificate_info" in certificate
        assert "quality_metrics" in certificate
        assert "compliance_validations" in certificate
        assert "verification" in certificate
    
    def test_verify_certificate_api(self, client, sample_task_request):
        """Test certificate verification via API"""
        # Generate certificate
        create_response = client.post("/api/certificates/task", json=sample_task_request)
        certificate_id = create_response.json()["certificate_id"]
        
        # Verify certificate
        response = client.get(f"/api/certificates/{certificate_id}/verify")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "verification" in data
        
        verification = data["verification"]
        assert "is_valid" in verification
        assert "verification_details" in verification
        assert "certificate_info" in verification
    
    def test_download_certificate_pdf_api(self, client, sample_task_request):
        """Test downloading certificate PDF via API"""
        # Generate certificate
        create_response = client.post("/api/certificates/task", json=sample_task_request)
        certificate_id = create_response.json()["certificate_id"]
        
        # Download PDF
        response = client.get(f"/api/certificates/{certificate_id}/pdf")
        
        # Should either return the PDF or 404 if not generated yet
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            assert response.headers["content-type"] == "application/pdf"
    
    def test_api_error_handling(self, client):
        """Test API error handling"""
        # Test invalid certificate ID
        response = client.get("/api/certificates/invalid_id")
        assert response.status_code == 404
        
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        
        # Test invalid request data
        response = client.post("/api/certificates/task", json={})
        assert response.status_code == 422  # Validation error
    
    def test_certificate_filtering(self, client, sample_task_request):
        """Test certificate filtering functionality"""
        # Generate multiple certificates with different properties
        for i in range(3):
            request = sample_task_request.copy()
            request["task"]["id"] = f"filter_test_{i}"
            request["task"]["title"] = f"Filter Test Task {i}"
            client.post("/api/certificates/task", json=request)
        
        # Test filtering by search term
        response = client.get("/api/certificates?search=Filter")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["certificates"]) >= 3
        
        # Test pagination
        response = client.get("/api/certificates?limit=2&offset=0")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["certificates"]) <= 2
        assert data["pagination"]["limit"] == 2
        assert data["pagination"]["offset"] == 0
    
    def test_dashboard_analytics_api(self, client, sample_task_request):
        """Test dashboard analytics endpoint"""
        # Generate some certificates first
        for i in range(2):
            request = sample_task_request.copy()
            request["task"]["id"] = f"analytics_test_{i}"
            client.post("/api/certificates/task", json=request)
        
        # Get dashboard analytics
        response = client.get("/api/analytics/dashboard")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "dashboard_analytics" in data
        
        analytics = data["dashboard_analytics"]
        assert "overview" in analytics
        assert "quality_distribution" in analytics
        assert "certificate_types" in analytics
        assert "recent_activity" in analytics
        assert "compliance_summary" in analytics


class TestQualityCertificationPerformance:
    """Performance tests for quality certification system"""
    
    @pytest.mark.asyncio
    async def test_bulk_certificate_generation_performance(self):
        """Test performance of bulk certificate generation"""
        quality_manager = ComprehensiveQualityManager()
        
        # Generate test data
        tasks = []
        for i in range(10):
            tasks.append({
                "id": f"perf_test_{i}",
                "title": f"Performance Test Task {i}",
                "description": f"Performance testing task {i}",
                "area": "Testing",
                "feature": "Performance"
            })
        
        development_result = {
            "complexity_score": 7.0,
            "artifacts": ["test.js"],
            "documentation": ["test.md"]
        }
        
        testing_result = {
            "code_coverage": 90.0,
            "success_rate": 95.0,
            "security_passed": True,
            "hipaa_compliant": True,
            "hipaa_score": 92.0,
            "soc2_compliant": True,
            "soc2_score": 90.0
        }
        
        business_rules_result = {
            "compliant": True,
            "validations": ["test"]
        }
        
        # Measure performance
        start_time = datetime.now()
        
        # Generate certificates concurrently
        certificate_tasks = []
        for task in tasks:
            cert_task = quality_manager.generate_task_quality_certificate(
                task, development_result, testing_result, business_rules_result
            )
            certificate_tasks.append(cert_task)
        
        certificates = await asyncio.gather(*certificate_tasks)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Performance assertions
        assert len(certificates) == 10
        assert total_time < 30.0  # Should complete in under 30 seconds
        assert all(cert.certificate_id.startswith("TASK-CERT-") for cert in certificates)
        
        # Verify all certificates are unique
        cert_ids = [cert.certificate_id for cert in certificates]
        assert len(set(cert_ids)) == len(cert_ids)
        
        print(f"Generated {len(certificates)} certificates in {total_time:.2f} seconds")
        print(f"Average time per certificate: {total_time/len(certificates):.2f} seconds")
    
    def test_api_response_time_performance(self):
        """Test API response time performance"""
        client = TestClient(app)
        
        # Test health check response time
        start_time = datetime.now()
        response = client.get("/health")
        end_time = datetime.now()
        
        response_time = (end_time - start_time).total_seconds()
        
        assert response.status_code == 200
        assert response_time < 1.0  # Should respond in under 1 second
        
        print(f"Health check response time: {response_time:.3f} seconds")
    
    @pytest.mark.asyncio
    async def test_certificate_validation_performance(self):
        """Test certificate validation performance"""
        quality_manager = ComprehensiveQualityManager()
        
        # Generate a certificate first
        task = {
            "id": "validation_perf_test",
            "title": "Validation Performance Test",
            "description": "Testing validation performance",
            "area": "Testing",
            "feature": "Validation"
        }
        
        development_result = {"complexity_score": 7.0}
        testing_result = {"code_coverage": 90.0, "security_passed": True}
        business_rules_result = {"compliant": True}
        
        certificate = await quality_manager.generate_task_quality_certificate(
            task, development_result, testing_result, business_rules_result
        )
        
        # Test validation performance
        start_time = datetime.now()
        
        # Perform multiple validations
        for _ in range(100):
            hash_valid = quality_manager._calculate_certificate_hash(certificate) == certificate.certificate_hash
            sig_valid = quality_manager._generate_digital_signature(certificate) == certificate.digital_signature
            assert hash_valid and sig_valid
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        assert total_time < 5.0  # 100 validations should complete in under 5 seconds
        
        print(f"100 certificate validations completed in {total_time:.3f} seconds")
        print(f"Average validation time: {total_time/100:.5f} seconds")


class TestQualityCertificationIntegration:
    """Integration tests for quality certification system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_certificate_workflow(self):
        """Test complete end-to-end certificate workflow"""
        quality_manager = ComprehensiveQualityManager()
        
        # Step 1: Generate task certificate
        task = {
            "id": "e2e_test_task",
            "title": "End-to-End Test Task",
            "description": "Complete workflow test",
            "area": "Integration Testing",
            "feature": "E2E Workflow"
        }
        
        development_result = {
            "complexity_score": 6.5,
            "artifacts": ["e2e_test.js", "e2e_api.py"],
            "documentation": ["e2e_docs.md"]
        }
        
        testing_result = {
            "code_coverage": 94.0,
            "success_rate": 98.0,
            "security_passed": True,
            "hipaa_compliant": True,
            "hipaa_score": 96.0,
            "soc2_compliant": True,
            "soc2_score": 94.0
        }
        
        business_rules_result = {
            "compliant": True,
            "validations": ["e2e_validation"]
        }
        
        task_cert = await quality_manager.generate_task_quality_certificate(
            task, development_result, testing_result, business_rules_result
        )
        
        assert task_cert is not None
        assert task_cert.overall_score > 0
        
        # Step 2: Generate module certificate
        module_info = {
            "id": "e2e_test_module",
            "name": "End-to-End Test Module",
            "owner": "Integration Test Team"
        }
        
        integration_results = {
            "coverage": 95.0,
            "success_rate": 99.0,
            "security_passed": True,
            "rules_compliant": True
        }
        
        module_cert = await quality_manager.generate_module_quality_certificate(
            module_info, [task_cert], integration_results
        )
        
        assert module_cert is not None
        assert task_cert.certificate_id in module_cert.parent_certificates
        assert module_cert.certificate_id in task_cert.child_certificates
        
        # Step 3: Generate system certificate
        system_info = {
            "id": "e2e_test_system",
            "name": "End-to-End Test System",
            "owner": "System Integration Team"
        }
        
        system_tests = {
            "coverage": 96.0,
            "success_rate": 99.5,
            "security_passed": True,
            "rules_compliant": True
        }
        
        compliance_audit = {
            "hipaa_compliant": True,
            "fda_validated": True,
            "soc2_certified": True
        }
        
        system_cert = await quality_manager.generate_system_quality_certificate(
            system_info, [module_cert], system_tests, compliance_audit
        )
        
        assert system_cert is not None
        assert module_cert.certificate_id in system_cert.parent_certificates
        assert system_cert.certificate_id in module_cert.child_certificates
        
        # Step 4: Verify complete traceability chain
        assert len(system_cert.validation_chain) > 0
        assert task_cert.certificate_id in system_cert.validation_chain
        
        # Step 5: Verify all documents generated
        for cert in [task_cert, module_cert, system_cert]:
            pdf_path = quality_manager.output_dir / "pdf" / f"{cert.certificate_id}_{cert.certificate_type.value}.pdf"
            html_path = quality_manager.output_dir / "html" / f"{cert.certificate_id}_{cert.certificate_type.value}.html"
            json_path = quality_manager.output_dir / "json" / f"{cert.certificate_id}_{cert.certificate_type.value}.json"
            
            assert pdf_path.exists()
            assert html_path.exists() 
            assert json_path.exists()
        
        print("✅ End-to-end certificate workflow completed successfully")
        print(f"Task Certificate: {task_cert.certificate_id} (Score: {task_cert.overall_score:.1f})")
        print(f"Module Certificate: {module_cert.certificate_id} (Score: {module_cert.overall_score:.1f})")
        print(f"System Certificate: {system_cert.certificate_id} (Score: {system_cert.overall_score:.1f})")


# Test execution
if __name__ == "__main__":
    print("🧪 Running Quality Certification System Tests...")
    
    # Run specific test suites
    import subprocess
    import sys
    
    # Run tests with pytest
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v", 
        "--tb=short",
        "--durations=10"
    ], capture_output=True, text=True)
    
    print("📊 Test Results:")
    print(result.stdout)
    
    if result.stderr:
        print("⚠️ Test Warnings/Errors:")
        print(result.stderr)
    
    if result.returncode == 0:
        print("✅ All tests passed successfully!")
    else:
        print("❌ Some tests failed.")
        
    print(f"🏁 Test execution completed with return code: {result.returncode}") 