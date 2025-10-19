#!/usr/bin/env python3
"""
Backblaze B2 DVC Setup Automation Script
Author: bigalex95
Date: 2025-10-19

This script automates the complete setup of DVC with Backblaze B2 storage.
It handles configuration, credential management, and verification.

Usage:
    python scripts/setup_b2_dvc.py --interactive
    python scripts/setup_b2_dvc.py --config config.json
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import getpass


class Colors:
    """ANSI color codes for terminal output"""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class B2DVCSetup:
    """Handles Backblaze B2 and DVC setup automation"""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.env_file = self.project_root / ".env"
        self.dvc_dir = self.project_root / ".dvc"
        self.config = {}

    def print_step(self, step: str, message: str):
        """Print formatted step message"""
        print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
        print(f"{Colors.BOLD}[{step}] {message}{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")

    def print_success(self, message: str):
        """Print success message"""
        print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")

    def print_error(self, message: str):
        """Print error message"""
        print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")

    def print_warning(self, message: str):
        """Print warning message"""
        print(f"{Colors.WARNING}⚠️  {message}{Colors.ENDC}")

    def print_info(self, message: str):
        """Print info message"""
        print(f"{Colors.OKCYAN}ℹ️  {message}{Colors.ENDC}")

    def run_command(self, cmd: list, capture_output: bool = False) -> Tuple[bool, str]:
        """Run shell command and return success status and output"""
        try:
            if capture_output:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return True, result.stdout
            else:
                subprocess.run(cmd, check=True)
                return True, ""
        except subprocess.CalledProcessError as e:
            return False, str(e)

    def check_dependencies(self) -> bool:
        """Check if required tools are installed"""
        self.print_step("STEP 1", "Checking Dependencies")

        dependencies = {
            "git": ["git", "--version"],
            "python": ["python", "--version"],
            "dvc": ["dvc", "version"],
        }

        all_installed = True

        for name, cmd in dependencies.items():
            success, output = self.run_command(cmd, capture_output=True)
            if success:
                self.print_success(f"{name} is installed: {output.strip()}")
            else:
                self.print_error(f"{name} is NOT installed")
                all_installed = False

        if not all_installed:
            self.print_error("Please install missing dependencies")
            return False

        return True

    def collect_b2_credentials(self, interactive: bool = True) -> Dict:
        """Collect B2 credentials from user or config file"""
        self.print_step("STEP 2", "Collecting Backblaze B2 Credentials")

        if interactive:
            print(
                f"{Colors.BOLD}Please provide your Backblaze B2 credentials:{Colors.ENDC}"
            )
            print(
                f"{Colors.OKCYAN}(You can find these in your B2 console under 'App Keys'){Colors.ENDC}\n"
            )

            bucket_name = input("Enter your B2 bucket name: ").strip()
            key_id = input("Enter your B2 Key ID (Access Key): ").strip()
            app_key = getpass.getpass(
                "Enter your B2 Application Key (Secret): "
            ).strip()

            print("\nAvailable B2 endpoints:")
            print("  1. s3.us-west-001.backblazeb2.com")
            print("  2. s3.us-west-002.backblazeb2.com")
            print("  3. s3.us-west-004.backblazeb2.com")
            print("  4. s3.eu-central-003.backblazeb2.com")
            print("  5. Custom endpoint")

            endpoint_choice = (
                input("\nSelect endpoint (1-5) [default: 3]: ").strip() or "3"
            )

            endpoint_map = {
                "1": "s3.us-west-001.backblazeb2.com",
                "2": "s3.us-west-002.backblazeb2.com",
                "3": "s3.us-west-004.backblazeb2.com",
                "4": "s3.eu-central-003.backblazeb2.com",
            }

            if endpoint_choice == "5":
                endpoint = input("Enter custom endpoint: ").strip()
            else:
                endpoint = endpoint_map.get(endpoint_choice, endpoint_map["3"])

            # Extract region from endpoint (e.g., us-west-004)
            region = endpoint.split(".")[1] if "." in endpoint else "us-west-004"

            credentials = {
                "bucket_name": bucket_name,
                "key_id": key_id,
                "application_key": app_key,
                "endpoint": f"https://{endpoint}",
                "region": region,
            }

        else:
            # Load from config file
            config_file = self.project_root / "b2_config.json"
            if not config_file.exists():
                self.print_error("Config file not found: b2_config.json")
                return {}

            with open(config_file, "r") as f:
                credentials = json.load(f)

        # Validate credentials
        if not all(credentials.values()):
            self.print_error("Invalid credentials provided")
            return {}

        self.config = credentials
        self.print_success("Credentials collected successfully")
        return credentials

    def create_env_file(self) -> bool:
        """Create .env file with B2 credentials"""
        self.print_step("STEP 3", "Creating .env File")

        env_content = f"""# Backblaze B2 Configuration
# Generated: 2025-10-19 13:23:42
# DO NOT COMMIT THIS FILE TO GIT!

# B2 Credentials
B2_BUCKET={self.config['bucket_name']}
B2_KEY_ID={self.config['key_id']}
B2_APPLICATION_KEY={self.config['application_key']}
B2_ENDPOINT={self.config['endpoint']}
B2_REGION={self.config['region']}

# AWS-compatible environment variables (for boto3/DVC)
AWS_ACCESS_KEY_ID={self.config['key_id']}
AWS_SECRET_ACCESS_KEY={self.config['application_key']}
AWS_ENDPOINT_URL={self.config['endpoint']}
AWS_DEFAULT_REGION={self.config['region']}

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL={self.config['endpoint']}

# DVC Configuration
DVC_REMOTE=myremote
"""

        try:
            with open(self.env_file, "w") as f:
                f.write(env_content)

            # Set file permissions (read/write for owner only)
            os.chmod(self.env_file, 0o600)

            self.print_success(f"Created .env file at: {self.env_file}")
            return True
        except Exception as e:
            self.print_error(f"Failed to create .env file: {e}")
            return False

    def update_gitignore(self) -> bool:
        """Update .gitignore to exclude sensitive files"""
        self.print_step("STEP 4", "Updating .gitignore")

        gitignore_file = self.project_root / ".gitignore"

        gitignore_entries = [
            "\n# Credentials and secrets",
            ".env",
            ".env.local",
            ".env.*.local",
            "b2_config.json",
            "",
            "# DVC",
            ".dvc/config.local",
            ".dvc/tmp",
            ".dvc/cache",
            "",
        ]

        try:
            # Read existing content
            existing_content = ""
            if gitignore_file.exists():
                with open(gitignore_file, "r") as f:
                    existing_content = f.read()

            # Add entries if not present
            with open(gitignore_file, "a") as f:
                for entry in gitignore_entries:
                    if entry.strip() and entry.strip() not in existing_content:
                        f.write(entry + "\n")

            self.print_success("Updated .gitignore")
            return True
        except Exception as e:
            self.print_error(f"Failed to update .gitignore: {e}")
            return False

    def initialize_dvc(self) -> bool:
        """Initialize DVC in the project"""
        self.print_step("STEP 5", "Initializing DVC")

        if self.dvc_dir.exists():
            self.print_warning("DVC already initialized")
            return True

        success, output = self.run_command(["dvc", "init"])
        if success:
            self.print_success("DVC initialized successfully")
            return True
        else:
            self.print_error(f"Failed to initialize DVC: {output}")
            return False

    def configure_dvc_remote(self) -> bool:
        """Configure DVC remote with B2"""
        self.print_step("STEP 6", "Configuring DVC Remote")

        remote_name = "myremote"
        remote_url = f"s3://{self.config['bucket_name']}/dvc-storage"

        # Remove existing remote if present
        self.run_command(["dvc", "remote", "remove", remote_name])

        # Add remote
        success, _ = self.run_command(
            ["dvc", "remote", "add", "-d", remote_name, remote_url]
        )

        if not success:
            self.print_error("Failed to add DVC remote")
            return False

        # Configure endpoint
        success, _ = self.run_command(
            [
                "dvc",
                "remote",
                "modify",
                remote_name,
                "endpointurl",
                self.config["endpoint"],
            ]
        )

        if not success:
            self.print_error("Failed to configure endpoint")
            return False

        # Configure credentials (local only, not committed)
        success, _ = self.run_command(
            [
                "dvc",
                "remote",
                "modify",
                "--local",
                remote_name,
                "access_key_id",
                self.config["key_id"],
            ]
        )

        if not success:
            self.print_error("Failed to configure access key")
            return False

        success, _ = self.run_command(
            [
                "dvc",
                "remote",
                "modify",
                "--local",
                remote_name,
                "secret_access_key",
                self.config["application_key"],
            ]
        )

        if not success:
            self.print_error("Failed to configure secret key")
            return False

        # Configure region
        success, _ = self.run_command(
            ["dvc", "remote", "modify", remote_name, "region", self.config["region"]]
        )

        if not success:
            self.print_warning("Failed to configure region (optional)")

        self.print_success(f"DVC remote '{remote_name}' configured successfully")
        return True

    def test_connection(self) -> bool:
        """Test connection to B2"""
        self.print_step("STEP 7", "Testing B2 Connection")

        try:
            import boto3
            from botocore.exceptions import ClientError

            # Create S3 client
            s3_client = boto3.client(
                "s3",
                endpoint_url=self.config["endpoint"],
                aws_access_key_id=self.config["key_id"],
                aws_secret_access_key=self.config["application_key"],
                region_name=self.config["region"],
            )

            # Try to list bucket contents
            self.print_info("Testing bucket access...")
            response = s3_client.list_objects_v2(
                Bucket=self.config["bucket_name"], MaxKeys=1
            )

            self.print_success(
                f"Successfully connected to bucket: {self.config['bucket_name']}"
            )
            return True

        except ClientError as e:
            self.print_error(f"Connection test failed: {e}")
            return False
        except ImportError:
            self.print_warning("boto3 not installed. Skipping connection test.")
            self.print_info("Install with: pip install boto3")
            return True

    def create_data_structure(self) -> bool:
        """Create recommended data directory structure"""
        self.print_step("STEP 8", "Creating Data Directory Structure")

        dirs_to_create = [
            "data/raw",
            "data/processed",
            "data/feedback",
            "data/temp",
        ]

        for dir_path in dirs_to_create:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)

            # Create .gitkeep to track empty directories
            gitkeep = full_path / ".gitkeep"
            gitkeep.touch()

            self.print_success(f"Created: {dir_path}/")

        return True

    def generate_helper_scripts(self) -> bool:
        """Generate helper scripts for DVC operations"""
        self.print_step("STEP 9", "Generating Helper Scripts")

        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        # DVC helper script
        dvc_helper = scripts_dir / "dvc_helper.sh"
        dvc_helper_content = """#!/bin/bash
# DVC Helper Script for Backblaze B2
# Auto-generated by setup_b2_dvc.py

set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
fi

case "$1" in
    pull)
        echo "📥 Pulling data from B2..."
        dvc pull
        ;;
    push)
        echo "📤 Pushing data to B2..."
        dvc push
        ;;
    status)
        echo "📊 DVC Status:"
        dvc status
        ;;
    check)
        echo "🔍 Testing B2 connection..."
        aws s3 ls s3://${B2_BUCKET}/ \\
            --endpoint-url ${B2_ENDPOINT}
        ;;
    *)
        echo "Usage: $0 {pull|push|status|check}"
        exit 1
        ;;
esac
"""

        try:
            with open(dvc_helper, "w") as f:
                f.write(dvc_helper_content)
            os.chmod(dvc_helper, 0o755)
            self.print_success(f"Created: {dvc_helper}")
        except Exception as e:
            self.print_error(f"Failed to create helper script: {e}")
            return False

        return True

    def print_summary(self):
        """Print setup summary and next steps"""
        self.print_step("SETUP COMPLETE", "Summary and Next Steps")

        print(f"{Colors.OKGREEN}{Colors.BOLD}")
        print("✅ Backblaze B2 and DVC setup completed successfully!")
        print(f"{Colors.ENDC}\n")

        print(f"{Colors.BOLD}Configuration Summary:{Colors.ENDC}")
        print(f"  • Bucket: {self.config['bucket_name']}")
        print(f"  • Endpoint: {self.config['endpoint']}")
        print(f"  • Region: {self.config['region']}")
        print(f"  • Environment file: {self.env_file}")
        print(f"  • DVC remote: myremote")

        print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
        print(f"{Colors.OKCYAN}")
        print("1. Track your dataset:")
        print("   $ dvc add data/raw/your_dataset")
        print("")
        print("2. Commit DVC files to Git:")
        print("   $ git add data/raw/your_dataset.dvc .dvc/config")
        print("   $ git commit -m 'Add dataset to DVC'")
        print("")
        print("3. Push data to B2:")
        print("   $ dvc push")
        print("")
        print("4. Push code to GitHub:")
        print("   $ git push origin main")
        print("")
        print("5. Team members can pull data:")
        print("   $ git clone <your-repo>")
        print("   $ dvc pull")
        print(f"{Colors.ENDC}")

        print(f"\n{Colors.BOLD}Helper Commands:{Colors.ENDC}")
        print("  • Pull data: ./scripts/dvc_helper.sh pull")
        print("  • Push data: ./scripts/dvc_helper.sh push")
        print("  • Check status: ./scripts/dvc_helper.sh status")
        print("  • Test connection: ./scripts/dvc_helper.sh check")

        print(f"\n{Colors.WARNING}{Colors.BOLD}⚠️  IMPORTANT:{Colors.ENDC}")
        print(f"{Colors.WARNING}")
        print("  • NEVER commit .env file to Git")
        print("  • NEVER commit .dvc/config.local")
        print("  • Share credentials securely with team members")
        print(f"{Colors.ENDC}")

    def run_setup(self, interactive: bool = True) -> bool:
        """Run complete setup process"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}")
        print("=" * 70)
        print(" Backblaze B2 + DVC Setup Automation")
        print(" Project: are-you-a-cat-mlops-pipeline")
        print(" Author: bigalex95")
        print("=" * 70)
        print(f"{Colors.ENDC}\n")

        # Step 1: Check dependencies
        if not self.check_dependencies():
            return False

        # Step 2: Collect credentials
        if not self.collect_b2_credentials(interactive):
            return False

        # Step 3: Create .env file
        if not self.create_env_file():
            return False

        # Step 4: Update .gitignore
        if not self.update_gitignore():
            return False

        # Step 5: Initialize DVC
        if not self.initialize_dvc():
            return False

        # Step 6: Configure DVC remote
        if not self.configure_dvc_remote():
            return False

        # Step 7: Test connection
        if not self.test_connection():
            self.print_warning("Connection test failed, but setup will continue")

        # Step 8: Create data structure
        if not self.create_data_structure():
            return False

        # Step 9: Generate helper scripts
        if not self.generate_helper_scripts():
            return False

        # Print summary
        self.print_summary()

        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Automate Backblaze B2 and DVC setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive setup
  python scripts/setup_b2_dvc.py --interactive
  
  # Non-interactive with config file
  python scripts/setup_b2_dvc.py --config b2_config.json
  
  # Specify project directory
  python scripts/setup_b2_dvc.py --interactive --project-dir /path/to/project
        """,
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (prompt for credentials)",
    )

    parser.add_argument(
        "--config", type=str, help="Path to config JSON file with B2 credentials"
    )

    parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Project root directory (default: current directory)",
    )

    args = parser.parse_args()

    if not args.interactive and not args.config:
        parser.error("Either --interactive or --config must be specified")

    # Run setup
    project_root = Path(args.project_dir).resolve()
    setup = B2DVCSetup(project_root=project_root)

    success = setup.run_setup(interactive=args.interactive)

    if success:
        sys.exit(0)
    else:
        print(f"\n{Colors.FAIL}Setup failed. Please check errors above.{Colors.ENDC}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
