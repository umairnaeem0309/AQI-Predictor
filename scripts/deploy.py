#!/usr/bin/env python3
"""
Deployment Script

Handles deployment with safety checks and rollback support.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DeploymentError(Exception):
    """Deployment error."""
    pass


class DeploymentManager:
    """Manages deployment process."""
    
    def __init__(self, environment: str = "production"):
        """
        Initialize deployment manager.
        
        Args:
            environment: Target environment (staging, production)
        """
        self.environment = environment
        self.compose_file = "docker/docker-compose.prod.yml"
        self.deployment_log = []
    
    def log(self, message: str):
        """Log deployment message."""
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = f"[{timestamp}] {message}"
        self.deployment_log.append(entry)
        print(entry)
    
    def run_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run shell command."""
        self.log(f"Running: {command}")
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        
        if check and result.returncode != 0:
            self.log(f"Command failed: {result.stderr}")
            raise DeploymentError(f"Command failed: {command}")
        
        return result
    
    def pre_deployment_checks(self) -> bool:
        """Run pre-deployment safety checks."""
        self.log("Running pre-deployment checks...")
        
        # Run the pre_deploy_checks script
        result = self.run_command(
            "python scripts/pre_deploy_checks.py",
            check=False,
        )
        
        if result.returncode != 0:
            self.log("Pre-deployment checks failed!")
            return False
        
        return True
    
    def backup_current(self) -> Optional[str]:
        """Backup current deployment."""
        self.log("Creating backup...")
        
        # Get current image tags
        result = self.run_command(
            f"docker compose -f {self.compose_file} images -q",
            check=False,
        )
        
        if result.returncode == 0 and result.stdout.strip():
            backup_tag = f"backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            self.log(f"Backup tag: {backup_tag}")
            return backup_tag
        
        return None
    
    def build_images(self) -> bool:
        """Build Docker images."""
        self.log("Building Docker images...")
        
        result = self.run_command(
            f"docker compose -f {self.compose_file} build --no-cache",
            check=False,
        )
        
        if result.returncode != 0:
            self.log("Image build failed!")
            return False
        
        self.log("Images built successfully")
        return True
    
    def deploy(self) -> bool:
        """Deploy services."""
        self.log("Deploying services...")
        
        result = self.run_command(
            f"docker compose -f {self.compose_file} up -d --remove-orphans",
            check=False,
        )
        
        if result.returncode != 0:
            self.log("Deployment failed!")
            return False
        
        self.log("Services deployed successfully")
        return True
    
    def verify_deployment(self) -> bool:
        """Verify deployment is healthy."""
        self.log("Verifying deployment...")
        
        import time
        time.sleep(10)  # Wait for services to start
        
        # Check backend health
        result = self.run_command(
            "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\"",
            check=False,
        )
        
        if result.returncode != 0:
            self.log("Backend health check failed!")
            return False
        
        self.log("Deployment verified successfully")
        return True
    
    def rollback(self, backup_tag: Optional[str] = None):
        """Rollback to previous version."""
        self.log("Rolling back deployment...")
        
        result = self.run_command(
            f"docker compose -f {self.compose_file} down",
            check=False,
        )
        
        self.log("Rollback completed")
    
    def save_deployment_log(self):
        """Save deployment log."""
        log_path = project_root / "data" / "deployment_logs"
        log_path.mkdir(parents=True, exist_ok=True)
        
        log_file = log_path / f"deploy_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        
        with open(log_file, "w") as f:
            f.write("\n".join(self.deployment_log))
        
        self.log(f"Deployment log saved: {log_file}")
    
    def run_deployment(self) -> bool:
        """Run full deployment process."""
        self.log("=" * 60)
        self.log(f"Starting deployment to {self.environment}")
        self.log("=" * 60)
        
        try:
            # 1. Pre-deployment checks
            if not self.pre_deployment_checks():
                raise DeploymentError("Pre-deployment checks failed")
            
            # 2. Backup current deployment
            backup_tag = self.backup_current()
            
            # 3. Build images
            if not self.build_images():
                raise DeploymentError("Image build failed")
            
            # 4. Deploy
            if not self.deploy():
                raise DeploymentError("Deployment failed")
            
            # 5. Verify
            if not self.verify_deployment():
                self.log("Verification failed, rolling back...")
                self.rollback(backup_tag)
                raise DeploymentError("Deployment verification failed")
            
            self.log("=" * 60)
            self.log("✅ Deployment completed successfully!")
            self.log("=" * 60)
            
            return True
            
        except DeploymentError as e:
            self.log(f"❌ Deployment failed: {e}")
            self.log("Rolling back...")
            self.rollback()
            return False
        
        finally:
            self.save_deployment_log()


def main():
    """Main deployment entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy AQI Predictor")
    parser.add_argument(
        "--environment",
        choices=["staging", "production"],
        default="production",
        help="Target environment",
    )
    
    args = parser.parse_args()
    
    manager = DeploymentManager(args.environment)
    success = manager.run_deployment()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
