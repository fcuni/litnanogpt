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
