#!/usr/bin/env python3
"""Test script to verify SWE-bench dataset support in Spider.

This script tests that the OpenHands scaffold can handle both
SWE-bench and SWE-smith dataset formats without errors.
"""

import sys
from pathlib import Path

# Add spider to path
spider_root = Path(__file__).parent.parent
sys.path.insert(0, str(spider_root))

# Test imports
try:
    from server.scaffolds.openhands_wrapper import (
        OpenHandsScaffold,
        OpenHandsScaffoldConfig,
        _get_workspace_dir_name,
        _get_instance_docker_image,
    )
    print("✓ Successfully imported Spider OpenHands scaffold")
except ImportError as e:
    print(f"✗ Failed to import Spider modules: {e}")
    print(f"  Make sure you're running from /home/ubuntu/spider/")
    sys.exit(1)

import pandas as pd


def test_workspace_dir_name():
    """Test workspace directory naming for both dataset formats."""
    print("\n=== Testing workspace directory naming ===")
    
    # Test SWE-bench format (with repo + version)
    swebench_instance = pd.Series({
        'instance_id': 'django__django-11099',
        'repo': 'django/django',
        'version': '3.0',
        'base_commit': 'abc123',
    })
    
    workspace_name = _get_workspace_dir_name(swebench_instance)
    expected = 'django__django__3.0'
    if workspace_name == expected:
        print(f"✓ SWE-bench workspace: {workspace_name}")
    else:
        print(f"✗ SWE-bench workspace: expected {expected}, got {workspace_name}")
    
    # Test SWE-smith format (with repo only, swesmith/ prefix)
    swesmith_instance = pd.Series({
        'instance_id': 'oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc',
        'repo': 'swesmith/oauthlib__oauthlib.1fd52536',
        'image_name': 'jyangballin/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536',
    })
    
    workspace_name = _get_workspace_dir_name(swesmith_instance)
    expected = 'oauthlib__oauthlib.1fd52536'
    if workspace_name == expected:
        print(f"✓ SWE-smith workspace: {workspace_name}")
    else:
        print(f"✗ SWE-smith workspace: expected {expected}, got {workspace_name}")


def test_docker_image_name():
    """Test Docker image name resolution for both dataset formats."""
    print("\n=== Testing Docker image name resolution ===")
    
    # Test SWE-bench format (no image_name, construct from instance_id)
    swebench_instance = pd.Series({
        'instance_id': 'django__django-11099',
        'repo': 'django/django',
        'version': '3.0',
        'base_commit': 'abc123',
    })
    
    image_name = _get_instance_docker_image(swebench_instance)
    if 'django' in image_name.lower() and '11099' in image_name:
        print(f"✓ SWE-bench image: {image_name}")
    else:
        print(f"✗ SWE-bench image unexpected: {image_name}")
    
    # Test SWE-smith format (has image_name field)
    swesmith_instance = pd.Series({
        'instance_id': 'oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc',
        'repo': 'swesmith/oauthlib__oauthlib.1fd52536',
        'image_name': 'jyangballin/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536',
    })
    
    image_name = _get_instance_docker_image(swesmith_instance)
    expected = 'jyangballin/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536'
    if image_name == expected:
        print(f"✓ SWE-smith image: {image_name}")
    else:
        print(f"✗ SWE-smith image: expected {expected}, got {image_name}")
    
    # Test with base_image_override
    override_image = 'nikolaik/python-nodejs:python3.12-nodejs22'
    image_name = _get_instance_docker_image(swebench_instance, base_image_override=override_image)
    if image_name == override_image:
        print(f"✓ Override image: {image_name}")
    else:
        print(f"✗ Override image: expected {override_image}, got {image_name}")


def test_config_creation():
    """Test that configs can be created for both dataset types."""
    print("\n=== Testing config creation ===")
    
    # Test SWE-bench config
    try:
        config = OpenHandsScaffoldConfig(
            output_dir=Path("./test_output"),
            dataset="princeton-nlp/SWE-bench",
            split="train",
            max_instances=5,
            agent_class="CodeActAgent",
            max_iterations=50,
            llm_model="gpt-4o",
            num_workers=1,
        )
        print(f"✓ SWE-bench config created: dataset={config.dataset}, split={config.split}")
    except Exception as e:
        print(f"✗ Failed to create SWE-bench config: {e}")
    
    # Test SWE-smith config
    try:
        config = OpenHandsScaffoldConfig(
            output_dir=Path("./test_output"),
            dataset="SWE-bench/SWE-smith",
            split="train",
            max_instances=5,
            agent_class="CodeActAgent",
            max_iterations=50,
            llm_model="gpt-4o",
            num_workers=1,
        )
        print(f"✓ SWE-smith config created: dataset={config.dataset}, split={config.split}")
    except Exception as e:
        print(f"✗ Failed to create SWE-smith config: {e}")
    
    # Test with generic image override
    try:
        config = OpenHandsScaffoldConfig(
            output_dir=Path("./test_output"),
            dataset="princeton-nlp/SWE-bench",
            split="train",
            max_instances=5,
            agent_class="CodeActAgent",
            max_iterations=50,
            llm_model="gpt-4o",
            num_workers=1,
            base_container_image_override="nikolaik/python-nodejs:python3.12-nodejs22",
        )
        print(f"✓ Generic image config created: override={config.base_container_image_override}")
    except Exception as e:
        print(f"✗ Failed to create generic image config: {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing SWE-bench Dataset Support in Spider")
    print("=" * 60)
    
    test_workspace_dir_name()
    test_docker_image_name()
    test_config_creation()
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the example configs in config/swe/")
    print("2. Choose a dataset: SWE-bench or SWE-smith")
    print("3. Ensure Docker images are available (or use base_container_image_override)")
    print("4. Run trajectory generation with a small batch first (max_instances=5)")
    print("\nFor more details, see:")
    print("- /home/ubuntu/SWE_DATASET_ANALYSIS.md")
    print("- /home/ubuntu/spider/config/swe/README_DATASETS.md")


if __name__ == "__main__":
    main()
