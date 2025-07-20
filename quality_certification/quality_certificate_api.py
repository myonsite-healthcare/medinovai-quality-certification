#!/usr/bin/env python3
"""
Quality Certificate API for MedinovAI
Comprehensive API endpoints for quality certificate management,
validation, reporting, and analytics
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import logging
from enum import Enum

# Import our quality manager
from comprehensive_quality_manager import (
    ComprehensiveQualityManager,
    QualityCertificate,
    QualityCertificateType,
    QualityLevel,
    ComplianceFramework
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MedinovAI Quality Certification API",
    description="Comprehensive Quality Certificate Management & Compliance Validation API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize quality manager
quality_manager = ComprehensiveQualityManager()


# PYDANTIC MODELS

class TaskCompletionRequest(BaseModel):
    """Request model for task completion quality certificate"""
    task: Dict[str, Any]
    development_result: Dict[str, Any]
    testing_result: Dict[str, Any]
    business_rules_result: Dict[str, Any]


class ModuleValidationRequest(BaseModel):
    """Request model for module validation quality certificate"""
    module_info: Dict[str, Any]
    task_certificate_ids: List[str]
    integration_results: Dict[str, Any]


class SystemValidationRequest(BaseModel):
    """Request model for system validation quality certificate"""
    system_info: Dict[str, Any]
    module_certificate_ids: List[str]
    system_tests: Dict[str, Any]
    compliance_audit: Dict[str, Any]


class CertificateFilter(BaseModel):
    """Filter model for certificate queries"""
    certificate_type: Optional[QualityCertificateType] = None
    quality_level: Optional[QualityLevel] = None
    issued_after: Optional[datetime] = None
    issued_before: Optional[datetime] = None
    subject_id: Optional[str] = None
    compliance_framework: Optional[ComplianceFramework] = None
    search_term: Optional[str] = None
    limit: int = Field(default=50, le=500)
    offset: int = Field(default=0, ge=0)


class ComplianceReportRequest(BaseModel):
    """Request model for compliance report generation"""
    frameworks: List[ComplianceFramework]
    date_range_start: datetime
    date_range_end: datetime
    include_remediation: bool = True
    include_analytics: bool = True


class QualityAnalyticsRequest(BaseModel):
    """Request model for quality analytics report"""
    time_period: str = Field(default="30d", regex="^(7d|30d|90d|1y)$")
    include_trends: bool = True
    include_comparisons: bool = True
    include_predictions: bool = False
    certificate_types: Optional[List[QualityCertificateType]] = None


# CERTIFICATE GENERATION ENDPOINTS

@app.post("/api/certificates/task", response_model=Dict[str, Any])
async def generate_task_quality_certificate(
    request: TaskCompletionRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate comprehensive quality certificate for completed task
    """
    try:
        logger.info(f"Generating task quality certificate for: {request.task.get('title', 'Unknown')}")
        
        # Generate certificate
        certificate = await quality_manager.generate_task_quality_certificate(
            request.task,
            request.development_result,
            request.testing_result,
            request.business_rules_result
        )
        
        # Schedule background tasks
        background_tasks.add_task(
            send_certificate_notification,
            certificate.certificate_id,
            certificate.issued_to,
            "Task Quality Certificate Generated"
        )
        
        return {
            "success": True,
            "certificate_id": certificate.certificate_id,
            "quality_level": certificate.quality_level.value,
            "overall_score": certificate.overall_score,
            "message": "Task quality certificate generated successfully",
            "pdf_url": f"/api/certificates/{certificate.certificate_id}/pdf",
            "html_url": f"/api/certificates/{certificate.certificate_id}/html",
            "verification_url": f"/api/certificates/{certificate.certificate_id}/verify"
        }
        
    except Exception as e:
        logger.error(f"Failed to generate task quality certificate: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Certificate generation failed: {str(e)}")


@app.post("/api/certificates/module", response_model=Dict[str, Any])
async def generate_module_quality_certificate(
    request: ModuleValidationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate comprehensive quality certificate for module validation
    """
    try:
        logger.info(f"Generating module quality certificate for: {request.module_info.get('name', 'Unknown')}")
        
        # Get task certificates
        task_certificates = []
        for cert_id in request.task_certificate_ids:
            if cert_id in quality_manager.certificates_db:
                task_certificates.append(quality_manager.certificates_db[cert_id])
        
        if not task_certificates:
            raise HTTPException(status_code=400, detail="No valid task certificates found")
        
        # Generate certificate
        certificate = await quality_manager.generate_module_quality_certificate(
            request.module_info,
            task_certificates,
            request.integration_results
        )
        
        # Schedule background tasks
        background_tasks.add_task(
            send_certificate_notification,
            certificate.certificate_id,
            certificate.issued_to,
            "Module Quality Certificate Generated"
        )
        
        return {
            "success": True,
            "certificate_id": certificate.certificate_id,
            "quality_level": certificate.quality_level.value,
            "overall_score": certificate.overall_score,
            "parent_certificates": len(task_certificates),
            "message": "Module quality certificate generated successfully",
            "pdf_url": f"/api/certificates/{certificate.certificate_id}/pdf",
            "html_url": f"/api/certificates/{certificate.certificate_id}/html",
            "verification_url": f"/api/certificates/{certificate.certificate_id}/verify"
        }
        
    except Exception as e:
        logger.error(f"Failed to generate module quality certificate: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Certificate generation failed: {str(e)}")


@app.post("/api/certificates/system", response_model=Dict[str, Any])
async def generate_system_quality_certificate(
    request: SystemValidationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate comprehensive system-wide quality certificate
    """
    try:
        logger.info(f"Generating system quality certificate for: {request.system_info.get('name', 'Unknown')}")
        
        # Get module certificates
        module_certificates = []
        for cert_id in request.module_certificate_ids:
            if cert_id in quality_manager.certificates_db:
                module_certificates.append(quality_manager.certificates_db[cert_id])
        
        if not module_certificates:
            raise HTTPException(status_code=400, detail="No valid module certificates found")
        
        # Generate certificate
        certificate = await quality_manager.generate_system_quality_certificate(
            request.system_info,
            module_certificates,
            request.system_tests,
            request.compliance_audit
        )
        
        # Schedule background tasks
        background_tasks.add_task(
            send_certificate_notification,
            certificate.certificate_id,
            certificate.issued_to,
            "System Quality Certificate Generated"
        )
        
        background_tasks.add_task(
            generate_system_analytics_report,
            certificate.certificate_id
        )
        
        return {
            "success": True,
            "certificate_id": certificate.certificate_id,
            "quality_level": certificate.quality_level.value,
            "overall_score": certificate.overall_score,
            "parent_certificates": len(module_certificates),
            "compliance_frameworks": len(certificate.compliance_validations),
            "message": "System quality certificate generated successfully",
            "pdf_url": f"/api/certificates/{certificate.certificate_id}/pdf",
            "html_url": f"/api/certificates/{certificate.certificate_id}/html",
            "verification_url": f"/api/certificates/{certificate.certificate_id}/verify",
            "analytics_url": f"/api/certificates/{certificate.certificate_id}/analytics"
        }
        
    except Exception as e:
        logger.error(f"Failed to generate system quality certificate: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Certificate generation failed: {str(e)}")


# CERTIFICATE RETRIEVAL ENDPOINTS

@app.get("/api/certificates", response_model=Dict[str, Any])
async def list_quality_certificates(
    certificate_type: Optional[str] = None,
    quality_level: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    search: Optional[str] = None
):
    """
    List quality certificates with filtering and pagination
    """
    try:
        certificates = list(quality_manager.certificates_db.values())
        
        # Apply filters
        if certificate_type:
            certificates = [c for c in certificates if c.certificate_type.value == certificate_type]
        
        if quality_level:
            certificates = [c for c in certificates if c.quality_level.value == quality_level]
        
        if search:
            search_lower = search.lower()
            certificates = [
                c for c in certificates 
                if search_lower in c.title.lower() or 
                   search_lower in c.subject_name.lower() or
                   search_lower in c.certificate_id.lower()
            ]
        
        # Sort by issue date (newest first)
        certificates.sort(key=lambda x: x.issue_date, reverse=True)
        
        # Pagination
        total = len(certificates)
        certificates = certificates[offset:offset + limit]
        
        # Convert to dict format for response
        certificate_list = []
        for cert in certificates:
            certificate_list.append({
                "certificate_id": cert.certificate_id,
                "title": cert.title,
                "certificate_type": cert.certificate_type.value,
                "quality_level": cert.quality_level.value,
                "overall_score": cert.overall_score,
                "subject_name": cert.subject_name,
                "issued_to": cert.issued_to,
                "issue_date": cert.issue_date,
                "expiry_date": cert.expiry_date,
                "test_coverage": cert.test_coverage,
                "test_success_rate": cert.test_success_rate,
                "security_scan_passed": cert.security_scan_passed,
                "business_rules_compliant": cert.business_rules_compliant,
                "compliance_summary": {
                    "total_frameworks": len(cert.compliance_validations),
                    "compliant_frameworks": len([v for v in cert.compliance_validations if v.compliant]),
                    "compliance_rate": (len([v for v in cert.compliance_validations if v.compliant]) / len(cert.compliance_validations) * 100) if cert.compliance_validations else 0
                }
            })
        
        return {
            "success": True,
            "certificates": certificate_list,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_next": offset + limit < total,
                "has_prev": offset > 0
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to list certificates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve certificates: {str(e)}")


@app.get("/api/certificates/{certificate_id}", response_model=Dict[str, Any])
async def get_quality_certificate_details(certificate_id: str):
    """
    Get detailed information about a specific quality certificate
    """
    try:
        if certificate_id not in quality_manager.certificates_db:
            raise HTTPException(status_code=404, detail="Certificate not found")
        
        certificate = quality_manager.certificates_db[certificate_id]
        
        # Convert to detailed dict format
        detailed_certificate = {
            "certificate_info": {
                "certificate_id": certificate.certificate_id,
                "title": certificate.title,
                "description": certificate.description,
                "certificate_type": certificate.certificate_type.value,
                "quality_level": certificate.quality_level.value,
                "overall_score": certificate.overall_score,
                "subject_id": certificate.subject_id,
                "subject_name": certificate.subject_name,
                "issued_by": certificate.issued_by,
                "issued_to": certificate.issued_to,
                "issue_date": certificate.issue_date,
                "expiry_date": certificate.expiry_date
            },
            "quality_metrics": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "max_value": metric.max_value,
                    "unit": metric.unit,
                    "category": metric.category,
                    "weight": metric.weight,
                    "threshold_pass": metric.threshold_pass,
                    "threshold_excellent": metric.threshold_excellent,
                    "status": "excellent" if metric.value >= metric.threshold_excellent else "pass" if metric.value >= metric.threshold_pass else "fail"
                }
                for metric in certificate.quality_metrics
            ],
            "compliance_validations": [
                {
                    "framework": validation.framework.value,
                    "compliant": validation.compliant,
                    "compliance_score": validation.compliance_score,
                    "validated_controls": validation.validated_controls,
                    "failed_controls": validation.failed_controls,
                    "remediation_required": validation.remediation_required,
                    "validation_date": validation.validation_date
                }
                for validation in certificate.compliance_validations
            ],
            "testing_summary": {
                "test_coverage": certificate.test_coverage,
                "test_success_rate": certificate.test_success_rate,
                "security_scan_passed": certificate.security_scan_passed,
                "performance_benchmarks": certificate.performance_benchmarks
            },
            "business_rules": {
                "compliant": certificate.business_rules_compliant,
                "tenant_validations": certificate.tenant_specific_validations
            },
            "artifacts": {
                "development": certificate.development_artifacts,
                "test": certificate.test_artifacts,
                "documentation": certificate.documentation_artifacts
            },
            "verification": {
                "certificate_hash": certificate.certificate_hash,
                "digital_signature": certificate.digital_signature,
                "qr_code_data": certificate.qr_code_data
            },
            "traceability": {
                "parent_certificates": certificate.parent_certificates,
                "child_certificates": certificate.child_certificates,
                "validation_chain": certificate.validation_chain,
                "audit_trail": certificate.audit_trail
            }
        }
        
        return {
            "success": True,
            "certificate": detailed_certificate
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get certificate details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve certificate: {str(e)}")


# CERTIFICATE VALIDATION ENDPOINTS

@app.get("/api/certificates/{certificate_id}/verify", response_model=Dict[str, Any])
async def verify_quality_certificate(certificate_id: str):
    """
    Verify the authenticity and integrity of a quality certificate
    """
    try:
        if certificate_id not in quality_manager.certificates_db:
            raise HTTPException(status_code=404, detail="Certificate not found")
        
        certificate = quality_manager.certificates_db[certificate_id]
        
        # Verify certificate hash
        calculated_hash = quality_manager._calculate_certificate_hash(certificate)
        hash_valid = calculated_hash == certificate.certificate_hash
        
        # Verify digital signature
        calculated_signature = quality_manager._generate_digital_signature(certificate)
        signature_valid = calculated_signature == certificate.digital_signature
        
        # Check expiry
        expiry_date = datetime.fromisoformat(certificate.expiry_date.replace('Z', '+00:00'))
        is_expired = datetime.now() > expiry_date
        
        # Validate compliance status
        compliance_valid = all(v.compliant for v in certificate.compliance_validations)
        
        # Overall validity
        is_valid = hash_valid and signature_valid and not is_expired and compliance_valid
        
        verification_result = {
            "certificate_id": certificate_id,
            "is_valid": is_valid,
            "verification_details": {
                "hash_verification": {
                    "valid": hash_valid,
                    "calculated_hash": calculated_hash,
                    "stored_hash": certificate.certificate_hash
                },
                "signature_verification": {
                    "valid": signature_valid,
                    "algorithm": "SHA-512"
                },
                "expiry_check": {
                    "expired": is_expired,
                    "expiry_date": certificate.expiry_date,
                    "days_until_expiry": (expiry_date - datetime.now()).days if not is_expired else 0
                },
                "compliance_check": {
                    "all_compliant": compliance_valid,
                    "compliant_frameworks": len([v for v in certificate.compliance_validations if v.compliant]),
                    "total_frameworks": len(certificate.compliance_validations)
                }
            },
            "certificate_info": {
                "title": certificate.title,
                "quality_level": certificate.quality_level.value,
                "overall_score": certificate.overall_score,
                "issued_by": certificate.issued_by,
                "issue_date": certificate.issue_date
            },
            "verification_timestamp": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "verification": verification_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to verify certificate: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Certificate verification failed: {str(e)}")


# CERTIFICATE DOCUMENT ENDPOINTS

@app.get("/api/certificates/{certificate_id}/pdf")
async def download_certificate_pdf(certificate_id: str):
    """
    Download PDF version of quality certificate
    """
    try:
        if certificate_id not in quality_manager.certificates_db:
            raise HTTPException(status_code=404, detail="Certificate not found")
        
        certificate = quality_manager.certificates_db[certificate_id]
        pdf_path = quality_manager.output_dir / "pdf" / f"{certificate_id}_{certificate.certificate_type.value}.pdf"
        
        if not pdf_path.exists():
            # Generate PDF if it doesn't exist
            await quality_manager._generate_pdf_certificate(certificate)
        
        return FileResponse(
            path=str(pdf_path),
            media_type="application/pdf",
            filename=f"quality_certificate_{certificate_id}.pdf"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF download failed: {str(e)}")


@app.get("/api/certificates/{certificate_id}/html")
async def download_certificate_html(certificate_id: str):
    """
    Download HTML version of quality certificate
    """
    try:
        if certificate_id not in quality_manager.certificates_db:
            raise HTTPException(status_code=404, detail="Certificate not found")
        
        certificate = quality_manager.certificates_db[certificate_id]
        html_path = quality_manager.output_dir / "html" / f"{certificate_id}_{certificate.certificate_type.value}.html"
        
        if not html_path.exists():
            # Generate HTML if it doesn't exist
            await quality_manager._generate_html_certificate(certificate)
        
        return FileResponse(
            path=str(html_path),
            media_type="text/html",
            filename=f"quality_certificate_{certificate_id}.html"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download HTML: {str(e)}")
        raise HTTPException(status_code=500, detail=f"HTML download failed: {str(e)}")


# REPORTING ENDPOINTS

@app.post("/api/reports/compliance", response_model=Dict[str, Any])
async def generate_compliance_report(
    request: ComplianceReportRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate comprehensive compliance report
    """
    try:
        report_id = str(uuid.uuid4())
        
        # Schedule background task for report generation
        background_tasks.add_task(
            generate_compliance_report_task,
            report_id,
            request.frameworks,
            request.date_range_start,
            request.date_range_end,
            request.include_remediation,
            request.include_analytics
        )
        
        return {
            "success": True,
            "report_id": report_id,
            "message": "Compliance report generation started",
            "status_url": f"/api/reports/{report_id}/status",
            "download_url": f"/api/reports/{report_id}/download"
        }
        
    except Exception as e:
        logger.error(f"Failed to generate compliance report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.post("/api/reports/quality-analytics", response_model=Dict[str, Any])
async def generate_quality_analytics_report(
    request: QualityAnalyticsRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate comprehensive quality analytics report
    """
    try:
        report_id = str(uuid.uuid4())
        
        # Schedule background task for report generation
        background_tasks.add_task(
            generate_quality_analytics_report_task,
            report_id,
            request.time_period,
            request.include_trends,
            request.include_comparisons,
            request.include_predictions,
            request.certificate_types
        )
        
        return {
            "success": True,
            "report_id": report_id,
            "message": "Quality analytics report generation started",
            "status_url": f"/api/reports/{report_id}/status",
            "download_url": f"/api/reports/{report_id}/download"
        }
        
    except Exception as e:
        logger.error(f"Failed to generate quality analytics report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


# ANALYTICS ENDPOINTS

@app.get("/api/analytics/dashboard", response_model=Dict[str, Any])
async def get_quality_dashboard_analytics():
    """
    Get comprehensive analytics for quality dashboard
    """
    try:
        certificates = list(quality_manager.certificates_db.values())
        
        # Calculate overall statistics
        total_certificates = len(certificates)
        avg_quality_score = sum(cert.overall_score for cert in certificates) / total_certificates if certificates else 0
        
        # Quality level distribution
        quality_levels = {}
        for cert in certificates:
            level = cert.quality_level.value
            quality_levels[level] = quality_levels.get(level, 0) + 1
        
        # Compliance rate calculation
        all_validations = []
        for cert in certificates:
            all_validations.extend(cert.compliance_validations)
        
        compliance_rate = (len([v for v in all_validations if v.compliant]) / len(all_validations) * 100) if all_validations else 0
        
        # Certificate type distribution
        cert_types = {}
        for cert in certificates:
            cert_type = cert.certificate_type.value
            cert_types[cert_type] = cert_types.get(cert_type, 0) + 1
        
        # Recent activity (last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_certificates = [
            cert for cert in certificates 
            if datetime.fromisoformat(cert.issue_date.replace('Z', '+00:00')) > thirty_days_ago
        ]
        
        return {
            "success": True,
            "dashboard_analytics": {
                "overview": {
                    "total_certificates": total_certificates,
                    "average_quality_score": round(avg_quality_score, 2),
                    "compliance_rate": round(compliance_rate, 2),
                    "recent_certificates": len(recent_certificates)
                },
                "quality_distribution": quality_levels,
                "certificate_types": cert_types,
                "recent_activity": [
                    {
                        "certificate_id": cert.certificate_id,
                        "title": cert.title,
                        "quality_level": cert.quality_level.value,
                        "overall_score": cert.overall_score,
                        "issue_date": cert.issue_date
                    }
                    for cert in sorted(recent_certificates, key=lambda x: x.issue_date, reverse=True)[:10]
                ],
                "compliance_summary": {
                    framework.value: {
                        "total": len([v for v in all_validations if v.framework == framework]),
                        "compliant": len([v for v in all_validations if v.framework == framework and v.compliant]),
                        "rate": round(len([v for v in all_validations if v.framework == framework and v.compliant]) / len([v for v in all_validations if v.framework == framework]) * 100, 2) if [v for v in all_validations if v.framework == framework] else 0
                    }
                    for framework in ComplianceFramework
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get dashboard analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analytics retrieval failed: {str(e)}")


# BACKGROUND TASKS

async def send_certificate_notification(certificate_id: str, recipient: str, subject: str):
    """Send notification about certificate generation"""
    logger.info(f"Sending notification for certificate {certificate_id} to {recipient}")
    # Implementation for sending notifications (email, Slack, etc.)


async def generate_system_analytics_report(certificate_id: str):
    """Generate detailed analytics report for system certificate"""
    logger.info(f"Generating analytics report for system certificate {certificate_id}")
    # Implementation for generating detailed analytics


async def generate_compliance_report_task(
    report_id: str,
    frameworks: List[ComplianceFramework],
    start_date: datetime,
    end_date: datetime,
    include_remediation: bool,
    include_analytics: bool
):
    """Background task for compliance report generation"""
    logger.info(f"Generating compliance report {report_id}")
    # Implementation for comprehensive compliance report generation


async def generate_quality_analytics_report_task(
    report_id: str,
    time_period: str,
    include_trends: bool,
    include_comparisons: bool,
    include_predictions: bool,
    certificate_types: Optional[List[QualityCertificateType]]
):
    """Background task for quality analytics report generation"""
    logger.info(f"Generating quality analytics report {report_id}")
    # Implementation for comprehensive quality analytics report generation


# HEALTH CHECK ENDPOINT

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Quality Certificate API",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "certificates_count": len(quality_manager.certificates_db)
    }


# ERROR HANDLERS

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 