"""
Skill Registry Module
=====================
Provides a simple registry for loading and managing OpenClaw skills.

This module loads skill configurations from a YAML file and provides
methods to query available skills, their configurations, and execution limits.

Usage:
    from skill_registry import SkillRegistry
    
    registry = SkillRegistry()
    skills = registry.list_skills()
    config = registry.get("hmm_regime_detect")
    is_enabled = registry.is_enabled("monte_carlo_paths")
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml


class SkillRegistry:
    """
    Manages the registration and configuration of OpenClaw skills.
    
    This class provides a simple interface for loading skill configurations
    from YAML files and querying their properties at runtime.
    
    Attributes:
        config_path: Path to the registry YAML file
        _skills: Dictionary of skill configurations
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the skill registry.
        
        Args:
            config_path: Optional path to a custom registry YAML file.
                        If not provided, defaults to registry_config.yaml
                        in the same directory as this module.
        """
        base = Path(__file__).parent
        
        # Use custom path if provided, otherwise default to registry_config.yaml
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = base / "registry_config.yaml"
        
        self._skills: Dict[str, Dict[str, Any]] = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load the skill registry from the YAML configuration file."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            self._skills = data.get("skills", {})
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Skill registry not found at: {self.config_path}"
            )
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in registry config: {e}")
    
    def list_skills(self) -> List[str]:
        """
        Get a list of all enabled skill names.
        
        Returns:
            List of skill names that are currently enabled
        """
        return [
            name 
            for name, config in self._skills.items() 
            if config.get("enabled", True)
        ]
    
    def list_all_skills(self) -> List[str]:
        """
        Get a list of all registered skill names (including disabled).
        
        Returns:
            List of all skill names in the registry
        """
        return list(self._skills.keys())
    
    def get(self, name: str) -> Dict[str, Any]:
        """
        Get the full configuration for a skill.
        
        Args:
            name: The skill name to retrieve
            
        Returns:
            Dictionary containing the skill's configuration
            
        Raises:
            KeyError: If the skill is not found in the registry
        """
        if name not in self._skills:
            raise KeyError(f"Skill not found: {name}")
        
        return self._skills[name]
    
    def is_enabled(self, name: str) -> bool:
        """
        Check if a skill is currently enabled.
        
        Args:
            name: The skill name to check
            
        Returns:
            True if the skill exists and is enabled, False otherwise
        """
        if name not in self._skills:
            return False
        
        return self._skills[name].get("enabled", True)
    
    def get_limits(self, name: str) -> Dict[str, Any]:
        """
        Get the resource limits for a skill.
        
        Args:
            name: The skill name
            
        Returns:
            Dictionary containing runtime and memory limits
        """
        config = self.get(name)
        return {
            "max_runtime_ms": config.get("max_runtime_ms", 5000),
            "max_memory_mb": config.get("max_memory_mb", 256),
        }
    
    def reload(self) -> None:
        """
        Reload the registry from disk.
        
        Useful for picking up changes to the configuration file
        without restarting the application.
        """
        self._load_registry()
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        enabled = len(self.list_skills())
        total = len(self.list_all_skills())
        return f"SkillRegistry({enabled}/{total} skills enabled)"


# Global singleton instance for convenience
_default_registry: Optional[SkillRegistry] = None


def get_registry() -> SkillRegistry:
    """
    Get the default global skill registry instance.
    
    Returns:
        The global SkillRegistry instance
    """
    global _default_registry
    
    if _default_registry is None:
        _default_registry = SkillRegistry()
    
    return _default_registry


def list_skills() -> List[str]:
    """
    Convenience function to list all enabled skills.
    
    Returns:
        List of enabled skill names
    """
    return get_registry().list_skills()


def get_skill(name: str) -> Dict[str, Any]:
    """
    Convenience function to get a skill configuration.
    
    Args:
        name: The skill name
        
    Returns:
        The skill configuration dictionary
    """
    return get_registry().get(name)


def is_skill_enabled(name: str) -> bool:
    """
    Convenience function to check if a skill is enabled.
    
    Args:
        name: The skill name
        
    Returns:
        True if enabled, False otherwise
    """
    return get_registry().is_enabled(name)
