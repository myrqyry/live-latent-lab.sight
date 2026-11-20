import sys

try:
    import obspython as obs
except ImportError:
    print("ERROR: 'obspython' module not found.")
    print("This script must be run inside OBS Studio (Tools -> Scripts) or with an environment that has 'obspython'.")
    sys.exit(1)

def test_source_lifecycle():
    print("Testing Source Lifecycle...")

    # Create the source
    source_id = "ai_image_generator"
    source_name = "Test Generator"
    source = obs.obs_source_create(source_id, source_name, None, None)

    if not source:
        print(f"FAILURE: Could not create source '{source_id}'")
        return

    print(f"Source '{source_name}' created successfully.")

    # Test Settings Update
    settings = obs.obs_data_create()
    test_prompt = "A futuristic city with flying cars"
    obs.obs_data_set_string(settings, "prompt", test_prompt)
    obs.obs_source_update(source, settings)

    # Verify the property persisted
    current_settings = obs.obs_source_get_settings(source)
    val = obs.obs_data_get_string(current_settings, "prompt")

    if val != test_prompt:
        print(f"FAILURE: Expected prompt '{test_prompt}', got '{val}'")
    else:
        print("SUCCESS: Settings updated and verified correctly")

    # Cleanup
    obs.obs_data_release(settings)
    obs.obs_data_release(current_settings)
    obs.obs_source_release(source)
    print("Source released.")

if __name__ == "__main__":
    test_source_lifecycle()
