from nanogpt.training.datamodules.datamodules_utils import HFDatasetSpec


def get_tiny_shakespeare_spec() -> HFDatasetSpec:
    return HFDatasetSpec(dataset_name="karpathy/tiny_shakespeare", feature_name="text", trust_remote_code=True)


def get_openwebtxt_10k_spec() -> HFDatasetSpec:
    return HFDatasetSpec(
        dataset_name="stas/openwebtext-10k",
        feature_name="text",
        valid_split_label=None,
        test_split_label=None,
        trust_remote_code=True
    )


_ALL_SPECS = {"tiny_shakespeare": get_tiny_shakespeare_spec, "openwebtxt-10k": get_openwebtxt_10k_spec}


def get_spec_by_name(name: str) -> HFDatasetSpec:
    if name not in _ALL_SPECS:
        raise ValueError(f"Spec name {name} not found. Available specs: {_ALL_SPECS.keys()}")

    return _ALL_SPECS[name]()
