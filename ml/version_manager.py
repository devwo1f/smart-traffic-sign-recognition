"""
Model Version Manager

Semantic versioning for model artifacts. Tracks accuracy, config,
timestamps, and supports rollback.
"""

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import config


class VersionManager:
    """Manages model versioning with metadata tracking."""

    def __init__(self, versions_file: Path | None = None):
        self.versions_file = versions_file or config.VERSIONS_FILE
        self.versions = self._load_versions()

    def _load_versions(self) -> list[dict]:
        """Load versions from JSON file."""
        if self.versions_file.exists():
            with open(self.versions_file, "r") as f:
                data = json.load(f)
                return data.get("versions", [])
        return []

    def _save_versions(self) -> None:
        """Save versions to JSON file."""
        data = {
            "versions": self.versions,
            "current": self.versions[-1]["version"] if self.versions else None,
            "total_versions": len(self.versions),
        }
        with open(self.versions_file, "w") as f:
            json.dump(data, f, indent=2)

    def _get_next_version(self) -> str:
        """Generate next semantic version."""
        if not self.versions:
            return "v1.0.0"

        last = self.versions[-1]["version"]
        parts = last.lstrip("v").split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        minor += 1
        return f"v{major}.{minor}.{patch}"

    def _compute_config_hash(self) -> str:
        """Hash the current training config for comparison."""
        config_str = json.dumps({
            "backbone": config.CLASSIFIER_BACKBONE,
            "img_size": config.CLASSIFIER_IMG_SIZE,
            "batch_size": config.CLASSIFIER_BATCH_SIZE,
            "lr": config.CLASSIFIER_LR,
            "epochs": config.CLASSIFIER_EPOCHS,
        }, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def create_version(
        self,
        accuracy: float,
        notes: str = "",
    ) -> dict:
        """
        Create a new model version.

        Archives current model artifacts and records metadata.
        """
        version = self._get_next_version()
        timestamp = datetime.now(timezone.utc).isoformat()

        # Create version directory
        version_dir = config.MODELS_DIR / "archive" / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy model artifacts
        artifacts = {}
        for artifact, key in [
            (config.CLASSIFIER_CHECKPOINT, "checkpoint"),
            (config.CLASSIFIER_ONNX, "onnx"),
            (config.CLASSIFIER_TRT, "tensorrt"),
        ]:
            if artifact.exists():
                dest = version_dir / artifact.name
                shutil.copy2(artifact, dest)
                artifacts[key] = str(dest)

        version_info = {
            "version": version,
            "timestamp": timestamp,
            "accuracy": accuracy,
            "config_hash": self._compute_config_hash(),
            "backbone": config.CLASSIFIER_BACKBONE,
            "artifacts": artifacts,
            "notes": notes,
        }

        self.versions.append(version_info)
        self._save_versions()

        print(f"🏷️  Created version {version}")
        print(f"   Accuracy: {100 * accuracy:.2f}%")
        print(f"   Archived to: {version_dir}")

        return version_info

    def list_versions(self) -> list[dict]:
        """List all model versions."""
        return self.versions

    def get_version(self, version: str) -> dict | None:
        """Get a specific version's metadata."""
        for v in self.versions:
            if v["version"] == version:
                return v
        return None

    def rollback(self, version: str) -> bool:
        """
        Rollback to a previous model version.

        Restores model artifacts from the version archive.
        """
        target = self.get_version(version)
        if target is None:
            print(f"❌ Version {version} not found.")
            return False

        artifacts = target.get("artifacts", {})

        for key, src_path in artifacts.items():
            src = Path(src_path)
            if not src.exists():
                print(f"   ⚠️  Artifact not found: {src}")
                continue

            if key == "checkpoint":
                dest = config.CLASSIFIER_CHECKPOINT
            elif key == "onnx":
                dest = config.CLASSIFIER_ONNX
            elif key == "tensorrt":
                dest = config.CLASSIFIER_TRT
            else:
                continue

            shutil.copy2(src, dest)
            print(f"   ✅ Restored {key}: {dest}")

        print(f"\n🔄 Rolled back to {version}")
        return True

    def compare_versions(self, v1: str, v2: str) -> dict | None:
        """Compare two model versions."""
        ver1 = self.get_version(v1)
        ver2 = self.get_version(v2)

        if ver1 is None or ver2 is None:
            print(f"❌ Version not found: {v1 if ver1 is None else v2}")
            return None

        comparison = {
            "versions": [v1, v2],
            "accuracy_diff": ver2["accuracy"] - ver1["accuracy"],
            "config_changed": ver1["config_hash"] != ver2["config_hash"],
            "time_diff": ver2["timestamp"] if ver1 and ver2 else None,
        }

        print(f"\n📊 Version Comparison: {v1} vs {v2}")
        print(f"   Accuracy: {100 * ver1['accuracy']:.2f}% → {100 * ver2['accuracy']:.2f}% "
              f"({'↑' if comparison['accuracy_diff'] > 0 else '↓'}{abs(100 * comparison['accuracy_diff']):.2f}%)")
        print(f"   Config changed: {'Yes' if comparison['config_changed'] else 'No'}")

        return comparison


def main() -> None:
    """CLI interface for version manager."""
    vm = VersionManager()

    if len(sys.argv) < 2:
        print("Usage: python version_manager.py <command> [args]")
        print("Commands: list, rollback <version>, compare <v1> <v2>")
        return

    command = sys.argv[1]

    if command == "list":
        versions = vm.list_versions()
        if not versions:
            print("No versions found.")
            return
        print(f"\n{'Version':<12s} {'Accuracy':>10s} {'Backbone':<20s} {'Date'}")
        print("-" * 60)
        for v in versions:
            date = v["timestamp"][:10]
            print(f"{v['version']:<12s} {100 * v['accuracy']:>9.2f}% {v['backbone']:<20s} {date}")

    elif command == "rollback" and len(sys.argv) >= 3:
        vm.rollback(sys.argv[2])

    elif command == "compare" and len(sys.argv) >= 4:
        vm.compare_versions(sys.argv[2], sys.argv[3])

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
