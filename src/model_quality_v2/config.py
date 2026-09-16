"""Fixed candidates and serializable configuration for Model Quality V2."""

from dataclasses import asdict, dataclass, replace


STAGE_NAMES = {
    1: "k_threshold",
    2: "money_transform",
    3: "pay_representation",
    4: "engineered_blocks",
    5: "group_weight",
}
K_CANDIDATES = (13, 19, 25, 31, 35, 51, 75, 101)
MONEY_TRANSFORMS = ("standard", "signed_log", "yeo_johnson", "robust")
PAY_REPRESENTATIONS = ("raw", "structured")
FEATURE_BLOCK_CANDIDATES = ((), ("A",), ("B",), ("A", "B"))
PAY_GROUP_WEIGHTS = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class V2Configuration:
    k: int = 25
    metric: str = "euclidean"
    weights: str = "distance"
    money_transform: str = "standard"
    pay_representation: str = "raw"
    feature_blocks: tuple = ()
    pay_group_weight: float = 1.0

    def __post_init__(self):
        if self.k < 1:
            raise ValueError("k must be positive")
        if self.metric != "euclidean" or self.weights != "distance":
            raise ValueError("V2 fixes Euclidean distance and distance weighting")
        if self.money_transform not in MONEY_TRANSFORMS:
            raise ValueError("Unknown money transform")
        if self.pay_representation not in PAY_REPRESENTATIONS:
            raise ValueError("Unknown PAY representation")
        blocks = tuple(self.feature_blocks)
        if any(block not in {"A", "B"} for block in blocks) or len(set(blocks)) != len(blocks):
            raise ValueError("Feature blocks must be a unique subset of A and B")
        object.__setattr__(self, "feature_blocks", tuple(sorted(blocks)))
        if self.pay_group_weight not in PAY_GROUP_WEIGHTS:
            raise ValueError("PAY group weight must be 0.5, 1.0, or 2.0")

    @property
    def configuration_id(self):
        blocks = "none" if not self.feature_blocks else "".join(self.feature_blocks)
        return (
            f"k{self.k}__money-{self.money_transform}__pay-{self.pay_representation}"
            f"__blocks-{blocks}__payw-{self.pay_group_weight:g}"
        )

    @property
    def representation_key(self):
        return (
            self.money_transform,
            self.pay_representation,
            self.feature_blocks,
            self.pay_group_weight,
        )

    def to_dict(self):
        result = asdict(self)
        result["feature_blocks"] = list(self.feature_blocks)
        result["configuration_id"] = self.configuration_id
        return result

    @classmethod
    def from_dict(cls, value):
        fields = {key: value[key] for key in (
            "k", "metric", "weights", "money_transform", "pay_representation",
            "feature_blocks", "pay_group_weight",
        )}
        fields["feature_blocks"] = tuple(fields["feature_blocks"])
        return cls(**fields)


def stage_candidates(stage, current):
    if stage == 1:
        candidates = [replace(current, k=k) for k in K_CANDIDATES]
    elif stage == 2:
        candidates = [replace(current, money_transform=name) for name in MONEY_TRANSFORMS]
    elif stage == 3:
        candidates = [replace(current, pay_representation=name) for name in PAY_REPRESENTATIONS]
    elif stage == 4:
        candidates = [replace(current, feature_blocks=blocks) for blocks in FEATURE_BLOCK_CANDIDATES]
    elif stage == 5:
        candidates = [replace(current, pay_group_weight=weight) for weight in PAY_GROUP_WEIGHTS]
    else:
        raise ValueError("stage must be 1 through 5")
    return list(dict.fromkeys(candidates))
