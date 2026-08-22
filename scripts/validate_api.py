#!/usr/bin/env python3
"""
API Validation Script

Validates API credentials and endpoints for OpenWeather and AQICN.
"""

import os
import sys
import json
import requests
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class APIValidationError(Exception):
    """API validation error."""
    pass


class APIValidator:
    """Validates API credentials and endpoints."""
    
    def __init__(self):
        self.results = {}
    
    def validate_openweather(self, api_key: str) -> dict:
        """
        Validate OpenWeather API.
        
        Args:
            api_key: OpenWeather API key
            
        Returns:
            Validation results
        """
        print("Validating OpenWeather API...")
        
        results = {
            "api": "openweather",
            "checks": {},
            "passed": True,
        }
        
        # Test current weather endpoint
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q=Karachi&appid={api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results["checks"]["weather_endpoint"] = {
                    "passed": True,
                    "status_code": 200,
                    "data_received": bool(data),
                }
            elif response.status_code == 401:
                results["checks"]["weather_endpoint"] = {
                    "passed": False,
                    "error": "Invalid API key",
                }
                results["passed"] = False
            else:
                results["checks"]["weather_endpoint"] = {
                    "passed": False,
                    "status_code": response.status_code,
                }
                results["passed"] = False
        except Exception as e:
            results["checks"]["weather_endpoint"] = {
                "passed": False,
                "error": str(e),
            }
            results["passed"] = False
        
        # Test air pollution endpoint
        try:
            # Karachi coordinates
            url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat=24.8607&lon=67.0011&appid={api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results["checks"]["pollution_endpoint"] = {
                    "passed": True,
                    "status_code": 200,
                    "data_received": bool(data),
                }
            else:
                results["checks"]["pollution_endpoint"] = {
                    "passed": False,
                    "status_code": response.status_code,
                }
                results["passed"] = False
        except Exception as e:
            results["checks"]["pollution_endpoint"] = {
                "passed": False,
                "error": str(e),
            }
            results["passed"] = False
        
        return results
    
    def validate_aqicn(self, api_key: str) -> dict:
        """
        Validate AQICN API.
        
        Args:
            api_key: AQICN API key
            
        Returns:
            Validation results
        """
        print("Validating AQICN API...")
        
        results = {
            "api": "aqicn",
            "checks": {},
            "passed": True,
        }
        
        # Test station data endpoint
        try:
            # Karachi station
            url = f"https://api.waqi.info/feed/karachi/?token={api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    aqi = data.get("data", {}).get("aqi")
                    results["checks"]["station_endpoint"] = {
                        "passed": True,
                        "status_code": 200,
                        "aqi_value": aqi,
                        "data_fresh": aqi is not None and aqi != "-",
                    }
                else:
                    results["checks"]["station_endpoint"] = {
                        "passed": False,
                        "error": data.get("data", "Unknown error"),
                    }
                    results["passed"] = False
            else:
                results["checks"]["station_endpoint"] = {
                    "passed": False,
                    "status_code": response.status_code,
                }
                results["passed"] = False
        except Exception as e:
            results["checks"]["station_endpoint"] = {
                "passed": False,
                "error": str(e),
            }
            results["passed"] = False
        
        return results
    
    def validate_all(self) -> dict:
        """
        Validate all APIs.
        
        Returns:
            Combined validation results
        """
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "apis": {},
            "all_passed": True,
        }
        
        # Validate OpenWeather
        openweather_key = os.getenv("OPENWEATHER_API_KEY")
        if openweather_key:
            results["apis"]["openweather"] = self.validate_openweather(openweather_key)
            if not results["apis"]["openweather"]["passed"]:
                results["all_passed"] = False
        else:
            results["apis"]["openweather"] = {
                "passed": False,
                "error": "API key not configured",
            }
            results["all_passed"] = False
        
        # Validate AQICN
        aqicn_key = os.getenv("AQICN_API_KEY")
        if aqicn_key:
            results["apis"]["aqicn"] = self.validate_aqicn(aqicn_key)
            if not results["apis"]["aqicn"]["passed"]:
                results["all_passed"] = False
        else:
            results["apis"]["aqicn"] = {
                "passed": False,
                "error": "API key not configured",
            }
            results["all_passed"] = False
        
        return results
    
    def save_results(self, results: dict, output_dir: Path = None):
        """Save validation results."""
        if output_dir is None:
            output_dir = project_root / "data" / "api_validation"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"validation_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        return output_file


def main():
    """Main validation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate API credentials")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    
    args = parser.parse_args()
    
    validator = APIValidator()
    results = validator.validate_all()
    
    # Print summary
    print("\n" + "=" * 60)
    print("API Validation Summary")
    print("=" * 60)
    
    for api_name, api_results in results["apis"].items():
        status = "✅ PASSED" if api_results["passed"] else "❌ FAILED"
        print(f"{api_name}: {status}")
        
        for check_name, check_results in api_results.get("checks", {}).items():
            check_status = "✓" if check_results.get("passed") else "✗"
            print(f"  {check_status} {check_name}")
    
    print("\n" + "=" * 60)
    if results["all_passed"]:
        print("✅ All API validations passed!")
    else:
        print("❌ Some API validations failed!")
    print("=" * 60)
    
    # Save results
    if args.save:
        validator.save_results(results)
    
    sys.exit(0 if results["all_passed"] else 1)


if __name__ == "__main__":
    main()
